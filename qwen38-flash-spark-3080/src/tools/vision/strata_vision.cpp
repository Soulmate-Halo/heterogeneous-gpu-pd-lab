// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N]
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj, model;
    bool gpu = false;
    int threads = 0, max_tokens = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> --model <model.gguf> [--gpu] [--threads N] "
                             "[--max-tokens N]\n");
        return 2;
    }
    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* text = llama_model_load_from_file(model.c_str(), mp);
    if (!text) { std::printf("ERR cannot open the text model %s\n", model.c_str()); std::fflush(stdout); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.print_timings = false;
    cp.warmup = false;
    if (threads > 0) cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::printf("ERR cannot load the vision encoder %s\n", mmproj.c_str());
        std::fflush(stdout);
        return 1;
    }
    // a vocab-only model reports no hparams, so the width comes from the projector: its output is the model's input
    int n_embd = 0;
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(mmproj.c_str(), gp);
        const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
        if (k >= 0) n_embd = (int) gguf_get_val_u32(gg, k);
        if (gg) gguf_free(gg);
    }
    if (n_embd <= 0) { std::printf("ERR the vision encoder has no projection_dim\n"); std::fflush(stdout); return 1; }
    // Warm up at the LARGEST picture before READY: the encoder's GPU work buffers are allocated now, not at the first
    // real picture.  The server starts this process before the engine, so the engine sizes its expert cache from
    // what is really left; allocating ~1 GB later, on a GPU the engine has filled, made Windows page GPU memory and
    // the engine crawl to a standstill.  (A square image well above any cap; mtmd scales it to the token limit.)
    {
        const uint32_t side = 2048;
        std::vector<unsigned char> rgb((size_t) side * side * 3, 128);
        mtmd_bitmap* bm = mtmd_bitmap_init(side, side, rgb.data());
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bms[1] = {bm};
        int warm_tokens = 0;
        if (bm && mtmd_tokenize(ctx, chunks, &txt, bms, 1) == 0) {
            for (size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
                const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE && mtmd_encode_chunk(ctx, ch) == 0)
                    warm_tokens = (int) mtmd_input_chunk_get_n_tokens(ch);
            }
        }
        std::fprintf(stderr, "strata-vision: warmed up at %d image tokens\n", warm_tokens);
        mtmd_input_chunks_free(chunks);
        if (bm) mtmd_bitmap_free(bm);
    }
    std::printf("READY %d\n", n_embd);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false,
                                                                            mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        const mtmd_input_chunk* ichunk = nullptr;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the image could not be preprocessed";
        for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) ichunk = ch;
        }
        if (err.empty() && !ichunk) err = "no image chunk";
        if (err.empty() && mtmd_encode_chunk(ctx, ichunk) != 0) err = "the vision encoder failed";
        if (err.empty()) {
            const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ichunk);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ichunk);
            // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
            const int nx = (int) last.x + 1, ny = (int) last.y + 1;
            if (nx * ny != n) err = "the image grid is not rectangular (" + std::to_string(n) + " tokens)";
            const float* embd = mtmd_get_output_embd(ctx);
            FILE* f = err.empty() ? std::fopen(out.c_str(), "wb") : nullptr;
            if (err.empty() && !f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    }
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
