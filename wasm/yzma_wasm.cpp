// yzma_wasm.cpp is the WebAssembly shim between llama.cpp and the yzma
// pkg/llamawasm Go package.
//
// A WebAssembly module cannot use dlopen or libffi, so the Go bindings that
// yzma uses on native platforms do not work here. The Go code instead calls
// this shim through JavaScript. The rules of the interface are:
//
//   - Every function uses scalar arguments only. No struct crosses the
//     boundary, so the Go code does not need the wasm32 layout of any
//     llama.cpp struct.
//   - Every object is a small positive int32 handle. The Go code never holds
//     a WebAssembly address of a live object.
//   - A negative return value is an error. Use yzma_last_error to read the
//     text of the error.
//
// Increase YZMA_ABI_VERSION each time you change this interface. The Go code
// compares the value at startup and refuses a module that does not match.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define YZMA_ABI_VERSION 9

// Error codes. These are also in the Go code.
enum {
    YZMA_OK             =  0,
    YZMA_ERR_GENERIC    = -1,
    YZMA_ERR_HANDLE     = -2,
    YZMA_ERR_ALLOC      = -3,
    YZMA_ERR_LOAD       = -4,
    YZMA_ERR_TOO_SMALL  = -5,
};

// YZMA_ERR_BAD_HANDLE is the error of a bad handle in a function whose normal
// result can also be negative. llama_tokenize returns the negative of the
// number of tokens that it needs, and a token identifier can be -1, so those
// functions cannot use YZMA_ERR_HANDLE.
#define YZMA_ERR_BAD_HANDLE (-1000000)

namespace {

std::string last_error;

void set_error(const char * fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    last_error = buf;
}

// table holds objects of one type and gives each one an int32 handle. Handle 0
// is never used, so 0 is always an invalid handle.
//
// A handle is borrowed if another object owns the item. yzma_sampler_chain_get
// makes one, because the chain frees the samplers in it. The shim must not free
// a borrowed item.
template <typename T>
struct table {
    std::vector<T *> items;
    std::vector<bool> borrowed;

    int add(T * item) {
        return insert(item, false);
    }

    int add_borrowed(T * item) {
        return insert(item, true);
    }

    int insert(T * item, bool other_owner) {
        for (size_t i = 0; i < items.size(); i++) {
            if (items[i] == nullptr) {
                items[i] = item;
                borrowed[i] = other_owner;
                return (int) i + 1;
            }
        }
        items.push_back(item);
        borrowed.push_back(other_owner);
        return (int) items.size();
    }

    T * get(int handle) const {
        if (handle < 1 || (size_t) handle > items.size()) {
            return nullptr;
        }
        return items[handle - 1];
    }

    bool is_borrowed(int handle) const {
        if (handle < 1 || (size_t) handle > items.size()) {
            return false;
        }
        return borrowed[handle - 1];
    }

    T * take(int handle) {
        T * item = get(handle);
        if (item != nullptr) {
            items[handle - 1] = nullptr;
            borrowed[handle - 1] = false;
        }
        return item;
    }
};

table<llama_model>              models;
table<llama_context>            contexts;
table<const llama_vocab>        vocabs;
table<llama_sampler>            samplers;
table<mtmd_context>             mtmd_contexts;
table<mtmd_bitmap>              bitmaps;
table<mtmd_input_chunks>        chunk_lists;

// split_strings makes a vector of pointers into a buffer that holds n strings
// with a NUL byte after each one. A pointer to an array of pointers cannot
// cross the boundary, thus the Go side sends a list of strings in this shape.
std::vector<const char *> split_strings(const char * blob, int n) {
    std::vector<const char *> out;
    if (blob == nullptr || n <= 0) {
        return out;
    }

    out.reserve((size_t) n);
    const char * p = blob;
    for (int i = 0; i < n; i++) {
        out.push_back(p);
        p += strlen(p) + 1;
    }
    return out;
}

// copy_text puts a string of llama.cpp into buf with a NUL byte after it. It
// gives the number of bytes of the text, or YZMA_ERR_TOO_SMALL if buf is too
// small.
int copy_text(const char * text, char * buf, int cap) {
    if (text == nullptr) {
        return 0;
    }

    const int n = (int) strlen(text);
    if (cap < n + 1) {
        return YZMA_ERR_TOO_SMALL;
    }
    memcpy(buf, text, (size_t) n + 1);
    return n;
}

// log_level holds the lowest level of message that goes to the console. A
// value more than GGML_LOG_LEVEL_ERROR stops all messages.
int log_level = GGML_LOG_LEVEL_WARN;

void log_callback(ggml_log_level level, const char * text, void * /* user_data */) {
    if ((int) level < log_level) {
        return;
    }
    fputs(text, stderr);
}

// check_nmse is the normalized mean squared error of test against ref, which is
// mse(ref, test) divided by mse(ref, 0). The same measure that the
// test-backend-ops program of llama.cpp uses.
double check_nmse(const float * ref, const float * test, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) ref[i] - (double) test[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    if (den == 0.0) {
        return num == 0.0 ? 0.0 : 1.0;
    }
    return num / den;
}

// The shape of the graph of the self test. A matrix multiply of f16 weights by
// f32 activations is the op that a language model spends most of its time in,
// and the WebGPU backend needs f16 shaders, so this covers the path that
// matters.
enum {
    CHECK_K = 256, // length of a row
    CHECK_M = 64,  // rows of the weights
    CHECK_N = 8,   // columns of the activations
    CHECK_OUT = CHECK_M * CHECK_N,
};

// check_run builds the graph on one backend, fills it with the given values,
// and writes the result into out. It returns false if any step fails.
bool check_run(ggml_backend_t backend, const ggml_fp16_t * wdata, const float * adata, float * out) {
    ggml_init_params ip = {};
    ip.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        set_error("self test cannot make a ggml context");
        return false;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, CHECK_K, CHECK_M);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CHECK_K, CHECK_N);
    ggml_tensor * r = ggml_mul_mat(ctx, w, a);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, r);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        set_error("self test cannot allocate the tensors of %s", ggml_backend_name(backend));
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(w, wdata, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, adata, 0, ggml_nbytes(a));

    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(r, out, 0, ggml_nbytes(r));
    } else {
        set_error("self test cannot compute the graph on %s", ggml_backend_name(backend));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

} // namespace

extern "C" {

int yzma_abi_version(void) {
    return YZMA_ABI_VERSION;
}

// yzma_last_error copies the text of the last error into buf and returns the
// number of bytes copied, without the terminating zero byte. The return value
// is YZMA_ERR_TOO_SMALL if buf is too small.
int yzma_last_error(char * buf, int cap) {
    const int n = (int) last_error.size();
    if (cap < n + 1) {
        return YZMA_ERR_TOO_SMALL;
    }
    memcpy(buf, last_error.c_str(), (size_t) n + 1);
    return n;
}

void yzma_log_set_verbosity(int level) {
    log_level = level;
    llama_log_set(log_callback, nullptr);
}

// yzma_gpu_device copies the name of the first device that is not the CPU into
// buf and returns the number of bytes copied. The return value is 0 if
// llama.cpp has no device other than the CPU, which is what a build for the CPU
// always reports.
//
// A page can ask for WebGPU and still land on the CPU: the backend of llama.cpp
// gives no device if the adapter of the browser has no support for f16 shaders.
// This call says what llama.cpp really has, and not what the browser has.
int yzma_gpu_device(char * buf, int cap) {
    const size_t count = ggml_backend_dev_count();

    for (size_t i = 0; i < count; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        const char * name = ggml_backend_dev_name(dev);
        if (name == nullptr) {
            continue;
        }

        const int n = (int) strlen(name);
        if (cap < n + 1) {
            return YZMA_ERR_TOO_SMALL;
        }
        memcpy(buf, name, (size_t) n + 1);
        return n;
    }

    return 0;
}

// yzma_backend_check tests the device that is not the CPU. It runs one small
// matrix multiply on that device and the same one on the CPU, then compares the
// two results. The return value is 0 if the device agrees with the CPU, 1 if it
// gives wrong values, and a negative error code if the test cannot run.
//
// Some drivers give a WebGPU adapter that llama.cpp accepts and that then
// computes nonsense. A model on such a device answers with random tokens of the
// vocabulary. No property of the answer tells this apart from a weak model, so
// the only sure test is a comparison against the CPU.
//
// Call this after yzma_backend_init and before a model loads. The return value
// of 0 also comes back for a build that has only the CPU, because then there is
// nothing to test.
int yzma_backend_check(void) {
    ggml_backend_dev_t gpu = nullptr;

    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            gpu = dev;
            break;
        }
    }
    if (gpu == nullptr) {
        return 0;
    }

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        set_error("self test has no CPU device to compare against");
        return YZMA_ERR_GENERIC;
    }

    // The values stay in [-1,1] and come from a fixed sequence, so every run of
    // the test gives the same numbers.
    std::vector<ggml_fp16_t> wdata(CHECK_K * CHECK_M);
    std::vector<float>       adata(CHECK_K * CHECK_N);
    uint32_t seed = 1234567u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (float) ((double) (seed >> 8) / (double) (1u << 23)) - 1.0f;
    };
    for (size_t i = 0; i < wdata.size(); i++) {
        wdata[i] = ggml_fp32_to_fp16(next());
    }
    for (size_t i = 0; i < adata.size(); i++) {
        adata[i] = next();
    }

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu, nullptr);
    if (gpu_backend == nullptr) {
        set_error("self test cannot start the %s backend", ggml_backend_dev_name(gpu));
        return YZMA_ERR_GENERIC;
    }
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu, nullptr);
    if (cpu_backend == nullptr) {
        set_error("self test cannot start the CPU backend");
        ggml_backend_free(gpu_backend);
        return YZMA_ERR_GENERIC;
    }

    std::vector<float> got(CHECK_OUT);
    std::vector<float> want(CHECK_OUT);

    int result = YZMA_ERR_GENERIC;
    if (check_run(cpu_backend, wdata.data(), adata.data(), want.data()) &&
        check_run(gpu_backend, wdata.data(), adata.data(), got.data())) {

        // A correct f16 multiply on a GPU differs from the CPU only in the last
        // bits, which gives an error near 1e-6. Noise gives an error near 1.
        // The limit sits far from both, so a slow or an odd but correct driver
        // does not raise a false alarm.
        // A value that is not a number makes the error not a number, and an
        // infinite value makes it infinite. Neither is less than the limit,
        // thus this one test covers those results as well.
        const double err = check_nmse(want.data(), got.data(), want.size());

        result = err < 1e-2 ? 0 : 1;
        if (result == 1) {
            set_error("the %s backend gives wrong values, error %g", ggml_backend_dev_name(gpu), err);
        }
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
    return result;
}

void yzma_backend_init(void) {
    llama_log_set(log_callback, nullptr);
    llama_backend_init();
}

void yzma_backend_free(void) {
    llama_backend_free();
}

//
// model
//

// yzma_model_load loads a model from a file in the filesystem of the module.
// llama_model_params has no use_mmap member: Emscripten has no mmap, so
// llama.cpp reads the file.
int yzma_model_load(const char * path, int n_gpu_layers) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(path, params);
    if (model == nullptr) {
        set_error("cannot load model from %s", path);
        return YZMA_ERR_LOAD;
    }
    return models.add(model);
}

void yzma_model_free(int model) {
    llama_model * m = models.take(model);
    if (m != nullptr) {
        llama_model_free(m);
    }
}

int yzma_model_get_vocab(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    if (vocab == nullptr) {
        set_error("model %d has no vocab", model);
        return YZMA_ERR_GENERIC;
    }
    return vocabs.add(vocab);
}

int yzma_model_n_embd(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_n_embd(m);
}

int yzma_model_n_ctx_train(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_n_ctx_train(m);
}

// The calls that describe a model. Each one gives YZMA_ERR_HANDLE for a bad
// handle, except the two whose normal result can be negative.

int yzma_model_n_embd_inp(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_embd_inp(m);
}

int yzma_model_n_embd_out(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_embd_out(m);
}

int yzma_model_n_layer(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_layer(m);
}

int yzma_model_n_layer_nextn(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_layer_nextn(m);
}

int yzma_model_n_head(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_head(m);
}

int yzma_model_n_head_kv(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_head_kv(m);
}

int yzma_model_n_swa(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_swa(m);
}

int yzma_model_n_cls_out(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_n_cls_out(m);
}

int yzma_model_ftype(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_ftype(m);
}

int yzma_model_has_encoder(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_has_encoder(m);
}

int yzma_model_has_decoder(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_has_decoder(m);
}

int yzma_model_is_recurrent(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_is_recurrent(m);
}

int yzma_model_is_hybrid(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_is_hybrid(m);
}

int yzma_model_is_diffusion(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_model_is_diffusion(m);
}

// A rope type and the start token of a decoder can be -1, thus a bad handle
// gives YZMA_ERR_BAD_HANDLE.
int yzma_model_rope_type(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_BAD_HANDLE;
    }
    return (int) llama_model_rope_type(m);
}

int yzma_model_decoder_start_token(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_model_decoder_start_token(m);
}

// A size and a number of parameters do not fit an int, thus these give a
// double. It holds every whole number to 2^53, which is far above the size of
// a model. A bad handle gives -1.
double yzma_model_size(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return -1;
    }
    return (double) llama_model_size(m);
}

double yzma_model_n_params(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return -1;
    }
    return (double) llama_model_n_params(m);
}

// A bad handle gives 0, which is not a scale that a model uses.
float yzma_model_rope_freq_scale_train(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return 0;
    }
    return llama_model_rope_freq_scale_train(m);
}

int yzma_model_desc(int model, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_desc(m, buf, (size_t) cap);
}

// yzma_model_chat_template copies the built-in chat template of the model into
// buf. The return value is 0 if the model has no template.
int yzma_model_chat_template(int model, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    const char * tmpl = llama_model_chat_template(m, nullptr);
    if (tmpl == nullptr) {
        return 0;
    }
    const int n = (int) strlen(tmpl);
    if (cap < n + 1) {
        return YZMA_ERR_TOO_SMALL;
    }
    memcpy(buf, tmpl, (size_t) n + 1);
    return n;
}

// yzma_chat_apply_template puts one message into the chat format of the model
// and copies the result into buf.
//
// One message is enough for a prompt with a question about an image, which is
// what the multimodal calls need. A chat with turns needs more than this.
int yzma_chat_apply_template(int model, const char * role, const char * content,
                             int add_assistant, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }

    const char * tmpl = llama_model_chat_template(m, nullptr);
    if (tmpl == nullptr) {
        set_error("model %d has no chat template", model);
        return YZMA_ERR_GENERIC;
    }

    llama_chat_message message = { role, content };
    const int32_t n = llama_chat_apply_template(tmpl, &message, 1, add_assistant != 0, buf, cap);
    if (n > cap) {
        return YZMA_ERR_TOO_SMALL;
    }
    if (n < 0) {
        set_error("llama_chat_apply_template returned %d", n);
        return YZMA_ERR_GENERIC;
    }
    return n;
}

//
// context
//

static int context_new(int model, int n_ctx, int n_batch, int n_ubatch, int n_threads,
                       int embeddings, int pooling_type, int n_seq_max, int no_perf) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }

    llama_context_params params = llama_context_default_params();
    if (n_seq_max > 0) {
        params.n_seq_max = (uint32_t) n_seq_max;
    }
    if (n_ctx > 0) {
        params.n_ctx = (uint32_t) n_ctx;
    }
    if (n_batch > 0) {
        params.n_batch = (uint32_t) n_batch;
    }
    if (n_ubatch > 0) {
        params.n_ubatch = (uint32_t) n_ubatch;
    }
    if (n_threads > 0) {
        params.n_threads       = n_threads;
        params.n_threads_batch = n_threads;
    }
    params.embeddings   = embeddings != 0;
    params.pooling_type = (enum llama_pooling_type) pooling_type;
    params.no_perf      = no_perf != 0;

    llama_context * ctx = llama_init_from_model(m, params);
    if (ctx == nullptr) {
        set_error("cannot make a context for model %d", model);
        return YZMA_ERR_LOAD;
    }
    return contexts.add(ctx);
}

// yzma_context_new makes a context that holds one sequence.
int yzma_context_new(int model, int n_ctx, int n_batch, int n_ubatch, int n_threads,
                     int embeddings, int pooling_type) {
    return context_new(model, n_ctx, n_batch, n_ubatch, n_threads, embeddings, pooling_type, 1, 1);
}

// yzma_context_new_seq makes a context that holds n_seq_max sequences. A batch
// that carries more than one sequence needs it.
int yzma_context_new_seq(int model, int n_ctx, int n_batch, int n_ubatch, int n_threads,
                         int embeddings, int pooling_type, int n_seq_max) {
    return context_new(model, n_ctx, n_batch, n_ubatch, n_threads, embeddings, pooling_type, n_seq_max, 1);
}

// yzma_context_new_ext also takes no_perf. A context with no_perf 0 measures
// the time of each batch, which yzma_perf_context gives.
int yzma_context_new_ext(int model, int n_ctx, int n_batch, int n_ubatch, int n_threads,
                         int embeddings, int pooling_type, int n_seq_max, int no_perf) {
    return context_new(model, n_ctx, n_batch, n_ubatch, n_threads, embeddings, pooling_type, n_seq_max, no_perf);
}

void yzma_context_free(int ctx) {
    llama_context * c = contexts.take(ctx);
    if (c != nullptr) {
        llama_free(c);
    }
}

int yzma_context_n_ctx(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_n_ctx(c);
}

int yzma_context_n_batch(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_n_batch(c);
}

int yzma_context_n_ubatch(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_n_ubatch(c);
}

int yzma_context_n_seq_max(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_n_seq_max(c);
}

int yzma_context_n_ctx_seq(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_n_ctx_seq(c);
}

int yzma_context_n_threads(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return llama_n_threads(c);
}

int yzma_context_n_threads_batch(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return llama_n_threads_batch(c);
}

int yzma_context_n_rs_seq(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_n_rs_seq(c);
}

// yzma_set_n_threads changes the threads of a context after it is made.
int yzma_set_n_threads(int ctx, int n_threads, int n_threads_batch) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_set_n_threads(c, n_threads, n_threads_batch);
    return YZMA_OK;
}

// yzma_context_pooling_type gives how the context pools the embeddings. A
// pooling type can be -1, thus a bad handle gives YZMA_ERR_BAD_HANDLE.
int yzma_context_pooling_type(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_BAD_HANDLE;
    }
    return (int) llama_pooling_type(c);
}

// yzma_set_embeddings says if a decode gives the embeddings in place of the
// logits.
int yzma_set_embeddings(int ctx, int embeddings) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_set_embeddings(c, embeddings != 0);
    return YZMA_OK;
}

// yzma_set_causal_attn says if the attention is causal. An embedding model
// needs the attention of every token to every other one.
int yzma_set_causal_attn(int ctx, int causal) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_set_causal_attn(c, causal != 0);
    return YZMA_OK;
}

// yzma_synchronize waits for the computation of the context to end.
int yzma_synchronize(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_synchronize(c);
    return YZMA_OK;
}

// memory_of gives the memory of a context, or nullptr with the error set.
static llama_memory_t memory_of(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return nullptr;
    }
    return llama_get_memory(c);
}

int yzma_memory_clear(int ctx, int clear_data) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    llama_memory_clear(mem, clear_data != 0);
    return YZMA_OK;
}

// yzma_memory_seq_rm removes the tokens of a sequence between p0 and p1. A
// negative p0 is the start of the sequence and a negative p1 is the end of it.
// The result is 1 if the memory removed the tokens and 0 if it cannot, which
// happens when the memory keeps only whole sequences.
int yzma_memory_seq_rm(int ctx, int seq_id, int p0, int p1) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    return llama_memory_seq_rm(mem, (llama_seq_id) seq_id, (llama_pos) p0, (llama_pos) p1) ? 1 : 0;
}

// yzma_memory_seq_cp copies the tokens of one sequence between p0 and p1 into
// another sequence.
int yzma_memory_seq_cp(int ctx, int seq_id_src, int seq_id_dst, int p0, int p1) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    llama_memory_seq_cp(mem, (llama_seq_id) seq_id_src, (llama_seq_id) seq_id_dst,
                        (llama_pos) p0, (llama_pos) p1);
    return YZMA_OK;
}

// yzma_memory_seq_keep removes every sequence except one.
int yzma_memory_seq_keep(int ctx, int seq_id) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    llama_memory_seq_keep(mem, (llama_seq_id) seq_id);
    return YZMA_OK;
}

// yzma_memory_seq_add moves the tokens of a sequence between p0 and p1 by
// delta positions. This is how a program shifts a context that is full.
int yzma_memory_seq_add(int ctx, int seq_id, int p0, int p1, int delta) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    llama_memory_seq_add(mem, (llama_seq_id) seq_id, (llama_pos) p0, (llama_pos) p1, (llama_pos) delta);
    return YZMA_OK;
}

// yzma_memory_seq_div divides the positions of the tokens of a sequence
// between p0 and p1 by d.
int yzma_memory_seq_div(int ctx, int seq_id, int p0, int p1, int d) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    llama_memory_seq_div(mem, (llama_seq_id) seq_id, (llama_pos) p0, (llama_pos) p1, d);
    return YZMA_OK;
}

// yzma_memory_seq_pos_min gives the smallest position of a sequence, and -1 if
// the sequence is empty. A position is a normal negative value here, thus a bad
// handle gives YZMA_ERR_BAD_HANDLE.
int yzma_memory_seq_pos_min(int ctx, int seq_id) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_BAD_HANDLE;
    }
    return (int) llama_memory_seq_pos_min(mem, (llama_seq_id) seq_id);
}

// yzma_memory_seq_pos_max gives the largest position of a sequence, and -1 if
// the sequence is empty.
int yzma_memory_seq_pos_max(int ctx, int seq_id) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_BAD_HANDLE;
    }
    return (int) llama_memory_seq_pos_max(mem, (llama_seq_id) seq_id);
}

// yzma_memory_can_shift tells if the memory accepts yzma_memory_seq_add.
int yzma_memory_can_shift(int ctx) {
    llama_memory_t mem = memory_of(ctx);
    if (mem == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    return llama_memory_can_shift(mem) ? 1 : 0;
}

//
// vocab
//

int yzma_tokenize(int vocab, const char * text, int text_len, int * out, int out_cap,
                  int add_special, int parse_special) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_tokenize(v, text, text_len, (llama_token *) out, out_cap,
                          add_special != 0, parse_special != 0);
}

int yzma_token_to_piece(int vocab, int token, char * buf, int cap, int lstrip, int special) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_token_to_piece(v, (llama_token) token, buf, cap, lstrip, special != 0);
}

int yzma_vocab_is_eog(int vocab, int token) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_is_eog(v, (llama_token) token) ? 1 : 0;
}

int yzma_vocab_bos(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_bos(v);
}

int yzma_vocab_eos(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_eos(v);
}

int yzma_vocab_n_tokens(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_n_tokens(v);
}

int yzma_vocab_get_add_bos(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_get_add_bos(v) ? 1 : 0;
}

// yzma_vocab_eot gives the token that ends a turn.
int yzma_vocab_eot(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_eot(v);
}

// yzma_vocab_sep gives the token that separates two sentences.
int yzma_vocab_sep(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_sep(v);
}

// yzma_vocab_nl gives the token of a new line.
int yzma_vocab_nl(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_nl(v);
}

// yzma_vocab_pad gives the token that fills a batch.
int yzma_vocab_pad(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_pad(v);
}

// yzma_vocab_mask gives the token that hides a position.
int yzma_vocab_mask(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_mask(v);
}

// yzma_vocab_fim_pre gives the token before the text of a fill in the middle prompt.
int yzma_vocab_fim_pre(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_fim_pre(v);
}

// yzma_vocab_fim_suf gives the token after the text of a fill in the middle prompt.
int yzma_vocab_fim_suf(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_fim_suf(v);
}

// yzma_vocab_fim_mid gives the token of the middle of a fill in the middle prompt.
int yzma_vocab_fim_mid(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_fim_mid(v);
}

// yzma_vocab_fim_pad gives the token that fills a fill in the middle prompt.
int yzma_vocab_fim_pad(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_fim_pad(v);
}

// yzma_vocab_fim_rep gives the token of the repository of a fill in the middle prompt.
int yzma_vocab_fim_rep(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_fim_rep(v);
}

// yzma_vocab_fim_sep gives the token that separates the files of a fill in the middle prompt.
int yzma_vocab_fim_sep(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_fim_sep(v);
}

int yzma_vocab_get_add_eos(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_get_add_eos(v) ? 1 : 0;
}

int yzma_vocab_get_add_sep(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_get_add_sep(v) ? 1 : 0;
}

int yzma_vocab_is_control(int vocab, int token) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_vocab_is_control(v, (llama_token) token) ? 1 : 0;
}

// yzma_vocab_get_attr gives the attributes of a token. Each value of
// llama_token_attr is 0 or more, thus YZMA_ERR_HANDLE is sufficient here.
int yzma_vocab_get_attr(int vocab, int token) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_vocab_get_attr(v, (llama_token) token);
}

// yzma_vocab_type gives the kind of the tokenizer. Each value of
// llama_vocab_type is 0 or more, thus YZMA_ERR_HANDLE is sufficient here.
int yzma_vocab_type(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_vocab_type(v);
}

// yzma_vocab_get_score gives the score of a token. A result of this shape
// cannot carry an error, thus a bad handle gives 0.0f.
float yzma_vocab_get_score(int vocab, int token) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return 0.0f;
    }
    return llama_vocab_get_score(v, (llama_token) token);
}

// yzma_vocab_get_text copies the text of a token into buf. The vocabulary owns
// the text of llama_vocab_get_text, thus this makes a copy.
//
// The text is the raw form in the vocabulary, which holds the marks of the
// tokenizer. Use yzma_token_to_piece for the text that a program prints.
int yzma_vocab_get_text(int vocab, int token, char * buf, int cap) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }
    return copy_text(llama_vocab_get_text(v, (llama_token) token), buf, cap);
}

// yzma_vocab_get_suppress_tokens copies the tokens that the model suppresses
// into out. It gives the number of tokens, or the negative of the number that
// out needs.
int yzma_vocab_get_suppress_tokens(int vocab, int * out, int cap) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_BAD_HANDLE;
    }

    int32_t n = 0;
    const llama_token * tokens = llama_vocab_get_suppress_tokens(v, &n);
    if (tokens == nullptr || n <= 0) {
        return 0;
    }
    if (cap < n) {
        return -n;
    }

    memcpy(out, tokens, (size_t) n * sizeof(llama_token));
    return n;
}

//
// inference
//

// yzma_decode runs one batch of tokens through the model. The positions of the
// tokens come from the state of the context, the same as llama_batch_get_one.
int yzma_decode(int ctx, const int * tokens, int n_tokens) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_batch batch = llama_batch_get_one((llama_token *) tokens, n_tokens);
    const int rc = llama_decode(c, batch);
    if (rc != 0) {
        set_error("llama_decode returned %d", rc);
    }
    return rc;
}

int yzma_encode(int ctx, const int * tokens, int n_tokens) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_batch batch = llama_batch_get_one((llama_token *) tokens, n_tokens);
    const int rc = llama_encode(c, batch);
    if (rc != 0) {
        set_error("llama_encode returned %d", rc);
    }
    return rc;
}

// run_batch makes a llama_batch out of the parallel arrays that the Go side
// wrote into the memory of the module, and then decodes or encodes it.
//
// llama_batch keeps seq_id as an array of pointers, one for each token, and a
// pointer to an array of pointers cannot cross the boundary. The Go side thus
// sends one flat array of n_tokens * n_seq_max identifiers, and this makes the
// pointers into it. The arrays stay the property of the Go side: llama.cpp
// reads them during the call and keeps none of them.
static int run_batch(int ctx, const int * tokens, const int * pos, const int * n_seq_id,
                     const int * seq_ids, int n_seq_max, const signed char * logits,
                     int n_tokens, bool encode) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    if (n_tokens <= 0 || n_seq_max <= 0) {
        set_error("a batch needs a minimum of one token and one sequence, got %d and %d",
                  n_tokens, n_seq_max);
        return YZMA_ERR_GENERIC;
    }

    // llama_batch_init makes room for one pointer more than the number of
    // tokens and leaves the last one empty. Do the same here.
    std::vector<llama_seq_id *> seq_ptrs((size_t) n_tokens + 1, nullptr);
    for (int i = 0; i < n_tokens; i++) {
        seq_ptrs[(size_t) i] = (llama_seq_id *) seq_ids + (size_t) i * (size_t) n_seq_max;
    }

    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.token    = (llama_token *) tokens;
    batch.pos      = (llama_pos *) pos;
    batch.n_seq_id = (int32_t *) n_seq_id;
    batch.seq_id   = seq_ptrs.data();
    batch.logits   = (int8_t *) logits;

    const int rc = encode ? llama_encode(c, batch) : llama_decode(c, batch);
    if (rc != 0) {
        set_error("llama_%s returned %d", encode ? "encode" : "decode", rc);
    }
    return rc;
}

// yzma_decode_batch runs a batch that carries the position, the sequences, and
// the logit flag of each token. yzma_decode is the same call for a batch of one
// sequence whose positions continue from the state of the context.
int yzma_decode_batch(int ctx, const int * tokens, const int * pos, const int * n_seq_id,
                      const int * seq_ids, int n_seq_max, const signed char * logits,
                      int n_tokens) {
    return run_batch(ctx, tokens, pos, n_seq_id, seq_ids, n_seq_max, logits, n_tokens, false);
}

int yzma_encode_batch(int ctx, const int * tokens, const int * pos, const int * n_seq_id,
                      const int * seq_ids, int n_seq_max, const signed char * logits,
                      int n_tokens) {
    return run_batch(ctx, tokens, pos, n_seq_id, seq_ids, n_seq_max, logits, n_tokens, true);
}

// yzma_get_embeddings_seq copies n float values of the embedding of a sequence
// into out.
int yzma_get_embeddings_seq(int ctx, int seq_id, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * embd = llama_get_embeddings_seq(c, (llama_seq_id) seq_id);
    if (embd == nullptr) {
        set_error("context %d has no embeddings for sequence %d", ctx, seq_id);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, embd, (size_t) n * sizeof(float));
    return n;
}

// yzma_get_embeddings_ith copies n float values of the embedding of one token
// into out. An i of -1 takes the last token that has an embedding.
int yzma_get_embeddings_ith(int ctx, int i, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * embd = llama_get_embeddings_ith(c, i);
    if (embd == nullptr) {
        set_error("context %d has no embedding for token %d", ctx, i);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, embd, (size_t) n * sizeof(float));
    return n;
}

// yzma_get_embeddings copies n float values of the embeddings of the last
// batch into out. The embeddings of the tokens follow each other.
int yzma_get_embeddings(int ctx, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * embd = llama_get_embeddings(c);
    if (embd == nullptr) {
        set_error("context %d has no embeddings", ctx);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, embd, (size_t) n * sizeof(float));
    return n;
}

// yzma_get_logits_ith copies n float values of the logits of one token into
// out. An i of -1 takes the last token that has logits.
int yzma_get_logits_ith(int ctx, int i, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * logits = llama_get_logits_ith(c, i);
    if (logits == nullptr) {
        set_error("context %d has no logits for token %d", ctx, i);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, logits, (size_t) n * sizeof(float));
    return n;
}

// yzma_get_logits copies n float values of the logits of the last batch into
// out. The logits of the tokens follow each other, thus n is the number of
// tokens with logits times the size of the vocabulary.
int yzma_get_logits(int ctx, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * logits = llama_get_logits(c);
    if (logits == nullptr) {
        set_error("context %d has no logits", ctx);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, logits, (size_t) n * sizeof(float));
    return n;
}

//
// sampling
//

static int add_sampler(llama_sampler * smpl, const char * name) {
    if (smpl == nullptr) {
        set_error("cannot make the %s sampler", name);
        return YZMA_ERR_ALLOC;
    }
    return samplers.add(smpl);
}

int yzma_sampler_chain_new(int no_perf) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = no_perf != 0;
    return add_sampler(llama_sampler_chain_init(params), "chain");
}

// yzma_sampler_chain_add gives the sampler to the chain. The chain then owns
// the sampler, so the handle of the sampler becomes invalid.
int yzma_sampler_chain_add(int chain, int smpl) {
    llama_sampler * c = samplers.get(chain);
    llama_sampler * s = samplers.get(smpl);
    if (c == nullptr || s == nullptr) {
        set_error("invalid sampler handle: chain %d, sampler %d", chain, smpl);
        return YZMA_ERR_HANDLE;
    }
    if (samplers.is_borrowed(smpl)) {
        set_error("sampler %d is in a chain already", smpl);
        return YZMA_ERR_HANDLE;
    }
    llama_sampler_chain_add(c, s);

    // The chain now owns the sampler and frees it, so the handle goes away and
    // the shim does not free the sampler again.
    samplers.take(smpl);

    return YZMA_OK;
}

int yzma_sampler_greedy(void) {
    return add_sampler(llama_sampler_init_greedy(), "greedy");
}

int yzma_sampler_dist(unsigned int seed) {
    return add_sampler(llama_sampler_init_dist((uint32_t) seed), "dist");
}

int yzma_sampler_temp(float t) {
    return add_sampler(llama_sampler_init_temp(t), "temp");
}

int yzma_sampler_top_k(int k) {
    return add_sampler(llama_sampler_init_top_k(k), "top-k");
}

int yzma_sampler_top_p(float p, int min_keep) {
    return add_sampler(llama_sampler_init_top_p(p, (size_t) min_keep), "top-p");
}

int yzma_sampler_min_p(float p, int min_keep) {
    return add_sampler(llama_sampler_init_min_p(p, (size_t) min_keep), "min-p");
}

int yzma_sampler_penalties(int n_vocab, int last_n, float repeat, float freq, float present) {
    return add_sampler(llama_sampler_init_penalties(n_vocab, last_n, repeat, freq, present), "penalties");
}

int yzma_sampler_typical(float p, int min_keep) {
    return add_sampler(llama_sampler_init_typical(p, (size_t) min_keep), "typical");
}

int yzma_sampler_xtc(float p, float t, int min_keep, unsigned int seed) {
    return add_sampler(llama_sampler_init_xtc(p, t, (size_t) min_keep, (uint32_t) seed), "xtc");
}

int yzma_sampler_top_n_sigma(float n) {
    return add_sampler(llama_sampler_init_top_n_sigma(n), "top-n-sigma");
}

int yzma_sampler_temp_ext(float t, float delta, float exponent) {
    return add_sampler(llama_sampler_init_temp_ext(t, delta, exponent), "temp-ext");
}

int yzma_sampler_mirostat(int n_vocab, unsigned int seed, float tau, float eta, int m) {
    return add_sampler(llama_sampler_init_mirostat(n_vocab, (uint32_t) seed, tau, eta, m), "mirostat");
}

int yzma_sampler_mirostat_v2(unsigned int seed, float tau, float eta) {
    return add_sampler(llama_sampler_init_mirostat_v2((uint32_t) seed, tau, eta), "mirostat-v2");
}

int yzma_sampler_adaptive_p(float target, float decay, unsigned int seed) {
    return add_sampler(llama_sampler_init_adaptive_p(target, decay, (uint32_t) seed), "adaptive-p");
}

int yzma_sampler_infill(int vocab) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }
    return add_sampler(llama_sampler_init_infill(v), "infill");
}

// yzma_sampler_grammar makes a sampler that permits only the text that a GBNF
// grammar describes.
int yzma_sampler_grammar(int vocab, const char * grammar, const char * root) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }
    return add_sampler(llama_sampler_init_grammar(v, grammar, root), "grammar");
}

// yzma_sampler_grammar_lazy makes a grammar sampler that starts only after a
// pattern or a token of the trigger appears.
//
// patterns holds n_patterns strings with a NUL byte after each one. See
// split_strings.
int yzma_sampler_grammar_lazy(int vocab, const char * grammar, const char * root,
                              const char * patterns, int n_patterns,
                              const int * trigger_tokens, int n_trigger_tokens) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }

    std::vector<const char *> pats = split_strings(patterns, n_patterns);
    if (n_trigger_tokens < 0) {
        n_trigger_tokens = 0;
    }

    return add_sampler(llama_sampler_init_grammar_lazy_patterns(
                           v, grammar, root,
                           pats.empty() ? nullptr : pats.data(), pats.size(),
                           n_trigger_tokens == 0 ? nullptr : (const llama_token *) trigger_tokens,
                           (size_t) n_trigger_tokens),
                       "lazy grammar");
}

// yzma_sampler_dry makes a DRY sampler. breakers holds n_breakers strings with
// a NUL byte after each one. See split_strings.
int yzma_sampler_dry(int vocab, float multiplier, float base, int allowed_length,
                     int penalty_last, const char * breakers, int n_breakers) {
    const llama_vocab * v = vocabs.get(vocab);
    if (v == nullptr) {
        set_error("invalid vocab handle %d", vocab);
        return YZMA_ERR_HANDLE;
    }

    std::vector<const char *> seq = split_strings(breakers, n_breakers);

    return add_sampler(llama_sampler_init_dry(v, multiplier, base, allowed_length, penalty_last,
                                              seq.empty() ? nullptr : seq.data(), seq.size()),
                       "dry");
}

// yzma_sampler_logit_bias makes a sampler that moves the logit of a token.
//
// llama_sampler_init_logit_bias takes an array of llama_logit_bias, which is a
// struct. Thus the tokens and the biases arrive as two arrays and this builds
// the array of structs.
int yzma_sampler_logit_bias(int n_vocab, const int * tokens, const float * biases, int n) {
    if (n < 0) {
        n = 0;
    }

    std::vector<llama_logit_bias> items((size_t) n);
    for (int i = 0; i < n; i++) {
        items[i].token = (llama_token) tokens[i];
        items[i].bias  = biases[i];
    }

    return add_sampler(llama_sampler_init_logit_bias(n_vocab, n, items.empty() ? nullptr : items.data()),
                       "logit-bias");
}

// yzma_sampler_name copies the name of a sampler into buf.
int yzma_sampler_name(int smpl, char * buf, int cap) {
    llama_sampler * s = samplers.get(smpl);
    if (s == nullptr) {
        set_error("invalid sampler handle %d", smpl);
        return YZMA_ERR_HANDLE;
    }
    return copy_text(llama_sampler_name(s), buf, cap);
}

// yzma_sampler_get_seed gives the seed of a sampler. The value of
// LLAMA_DEFAULT_SEED is 0xFFFFFFFF, thus the Go side reads the result as a
// uint32.
int yzma_sampler_get_seed(int smpl) {
    llama_sampler * s = samplers.get(smpl);
    if (s == nullptr) {
        set_error("invalid sampler handle %d", smpl);
        return YZMA_ERR_HANDLE;
    }
    return (int) llama_sampler_get_seed(s);
}

// yzma_sampler_clone makes a copy of a sampler. The caller owns the copy.
int yzma_sampler_clone(int smpl) {
    llama_sampler * s = samplers.get(smpl);
    if (s == nullptr) {
        set_error("invalid sampler handle %d", smpl);
        return YZMA_ERR_HANDLE;
    }
    return add_sampler(llama_sampler_clone(s), "clone");
}

int yzma_sampler_chain_n(int chain) {
    llama_sampler * c = samplers.get(chain);
    if (c == nullptr) {
        set_error("invalid sampler handle %d", chain);
        return YZMA_ERR_HANDLE;
    }
    return llama_sampler_chain_n(c);
}

// yzma_sampler_chain_get gives a handle to the sampler at position i of a
// chain. The chain keeps the sampler, thus the handle is borrowed and
// yzma_sampler_free of it frees nothing.
int yzma_sampler_chain_get(int chain, int i) {
    llama_sampler * c = samplers.get(chain);
    if (c == nullptr) {
        set_error("invalid sampler handle %d", chain);
        return YZMA_ERR_HANDLE;
    }

    llama_sampler * s = llama_sampler_chain_get(c, i);
    if (s == nullptr) {
        set_error("chain %d has no sampler at %d", chain, i);
        return YZMA_ERR_HANDLE;
    }
    return samplers.add_borrowed(s);
}

// yzma_sampler_chain_remove takes the sampler at position i out of a chain. The
// caller then owns the sampler.
int yzma_sampler_chain_remove(int chain, int i) {
    llama_sampler * c = samplers.get(chain);
    if (c == nullptr) {
        set_error("invalid sampler handle %d", chain);
        return YZMA_ERR_HANDLE;
    }

    llama_sampler * s = llama_sampler_chain_remove(c, i);
    if (s == nullptr) {
        set_error("chain %d has no sampler at %d", chain, i);
        return YZMA_ERR_HANDLE;
    }
    return samplers.add(s);
}

int yzma_sampler_sample(int smpl, int ctx, int idx) {
    llama_sampler * s = samplers.get(smpl);
    llama_context * c = contexts.get(ctx);
    if (s == nullptr || c == nullptr) {
        set_error("invalid handle: sampler %d, context %d", smpl, ctx);
        return YZMA_ERR_HANDLE;
    }
    return llama_sampler_sample(s, c, idx);
}

int yzma_sampler_accept(int smpl, int token) {
    llama_sampler * s = samplers.get(smpl);
    if (s == nullptr) {
        set_error("invalid sampler handle %d", smpl);
        return YZMA_ERR_HANDLE;
    }
    llama_sampler_accept(s, (llama_token) token);
    return YZMA_OK;
}

int yzma_sampler_reset(int smpl) {
    llama_sampler * s = samplers.get(smpl);
    if (s == nullptr) {
        set_error("invalid sampler handle %d", smpl);
        return YZMA_ERR_HANDLE;
    }
    llama_sampler_reset(s);
    return YZMA_OK;
}

// yzma_sampler_free frees a sampler. A borrowed handle, which
// yzma_sampler_chain_get gives, goes away and the sampler stays, because the
// chain owns it.
void yzma_sampler_free(int smpl) {
    const bool owned = !samplers.is_borrowed(smpl);
    llama_sampler * s = samplers.take(smpl);
    if (s != nullptr && owned) {
        llama_sampler_free(s);
    }
}

//
// backend sampling
//
// llama.cpp can put the sampler in the compute graph, thus the backend gives
// the token and the values that made it. This is experimental in llama.cpp.
//

// yzma_set_sampler attaches a sampler to a sequence of a context. It gives 1
// when the backend takes the sampler and 0 when it does not.
int yzma_set_sampler(int ctx, int seq_id, int smpl) {
    llama_context * c = contexts.get(ctx);
    llama_sampler * s = samplers.get(smpl);
    if (c == nullptr || s == nullptr) {
        set_error("invalid handle: context %d, sampler %d", ctx, smpl);
        return YZMA_ERR_HANDLE;
    }
    return llama_set_sampler(c, (llama_seq_id) seq_id, s) ? 1 : 0;
}

// yzma_get_sampled_token_ith gives the token that the backend sampled for the
// output at i. A token can be -1, thus a bad handle gives
// YZMA_ERR_BAD_HANDLE.
int yzma_get_sampled_token_ith(int ctx, int i) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_BAD_HANDLE;
    }
    return llama_get_sampled_token_ith(c, i);
}

// sampled_count gives the number of values of one of the three arrays that
// the backend sampler makes.
static int sampled_count(int ctx, int i, uint32_t (*count)(llama_context *, int32_t)) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    return (int) count(c, i);
}

int yzma_get_sampled_probs_count_ith(int ctx, int i) {
    return sampled_count(ctx, i, llama_get_sampled_probs_count_ith);
}

int yzma_get_sampled_logits_count_ith(int ctx, int i) {
    return sampled_count(ctx, i, llama_get_sampled_logits_count_ith);
}

int yzma_get_sampled_candidates_count_ith(int ctx, int i) {
    return sampled_count(ctx, i, llama_get_sampled_candidates_count_ith);
}

// yzma_get_sampled_probs_ith copies n probabilities of the output at i into
// out. The count call gives n.
int yzma_get_sampled_probs_ith(int ctx, int i, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * probs = llama_get_sampled_probs_ith(c, i);
    if (probs == nullptr) {
        set_error("context %d has no sampled probabilities for output %d", ctx, i);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, probs, (size_t) n * sizeof(float));
    return n;
}

// yzma_get_sampled_logits_ith copies n logits of the output at i into out.
int yzma_get_sampled_logits_ith(int ctx, int i, float * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const float * logits = llama_get_sampled_logits_ith(c, i);
    if (logits == nullptr) {
        set_error("context %d has no sampled logits for output %d", ctx, i);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, logits, (size_t) n * sizeof(float));
    return n;
}

// yzma_get_sampled_candidates_ith copies n token identifiers of the output at
// i into out.
int yzma_get_sampled_candidates_ith(int ctx, int i, int * out, int n) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const llama_token * tokens = llama_get_sampled_candidates_ith(c, i);
    if (tokens == nullptr) {
        set_error("context %d has no sampled candidates for output %d", ctx, i);
        return YZMA_ERR_GENERIC;
    }
    memcpy(out, tokens, (size_t) n * sizeof(int));
    return n;
}

//
// multimodal
//
// These calls follow the mtmd library of llama.cpp. A program makes a bitmap
// from the pixels of an image, puts the text and the bitmaps into a list of
// chunks, and runs the chunks through the model. After that the generation loop
// is the same as it is for text alone.
//
// The pixels must be RGB, three bytes for each one, with no padding between the
// rows. A browser gets them from a canvas, so no image library goes into this
// build.
//

// yzma_mtmd_init_from_file loads the projector of a multimodal model.
int yzma_mtmd_init_from_file(const char * mmproj_path, int model, int n_threads, int use_gpu,
                            int image_min_tokens, int image_max_tokens) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }

    mtmd_context_params params = mtmd_context_params_default();
    params.print_timings = false;
    params.use_gpu       = use_gpu != 0;

    // The default of mtmd is four threads whatever the machine has, and putting
    // an image through a projector is the slowest part of an answer, so a caller
    // that knows the number of cores should say so.
    if (n_threads > 0) {
        params.n_threads = n_threads;
    }

    // A model with a resolution that changes makes more tokens for a larger
    // image. Fewer tokens is less work.
    if (image_min_tokens > 0) {
        params.image_min_tokens = image_min_tokens;
    }
    if (image_max_tokens > 0) {
        params.image_max_tokens = image_max_tokens;
    }

    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, m, params);
    if (mctx == nullptr) {
        set_error("cannot load the projector from %s", mmproj_path);
        return YZMA_ERR_LOAD;
    }
    return mtmd_contexts.add(mctx);
}

void yzma_mtmd_free(int mctx) {
    mtmd_context * c = mtmd_contexts.take(mctx);
    if (c != nullptr) {
        mtmd_free(c);
    }
}

int yzma_mtmd_support_vision(int mctx) {
    mtmd_context * c = mtmd_contexts.get(mctx);
    if (c == nullptr) {
        set_error("invalid mtmd handle %d", mctx);
        return YZMA_ERR_HANDLE;
    }
    return mtmd_support_vision(c) ? 1 : 0;
}

int yzma_mtmd_support_audio(int mctx) {
    mtmd_context * c = mtmd_contexts.get(mctx);
    if (c == nullptr) {
        set_error("invalid mtmd handle %d", mctx);
        return YZMA_ERR_HANDLE;
    }
    return mtmd_support_audio(c) ? 1 : 0;
}

// yzma_mtmd_get_marker copies the marker that stands for a piece of media in
// the text of a prompt.
int yzma_mtmd_get_marker(int mctx, char * buf, int cap) {
    mtmd_context * c = mtmd_contexts.get(mctx);
    if (c == nullptr) {
        set_error("invalid mtmd handle %d", mctx);
        return YZMA_ERR_HANDLE;
    }

    const char * marker = mtmd_get_marker(c);
    if (marker == nullptr) {
        return 0;
    }

    const int n = (int) strlen(marker);
    if (cap < n + 1) {
        return YZMA_ERR_TOO_SMALL;
    }
    memcpy(buf, marker, (size_t) n + 1);
    return n;
}

// yzma_mtmd_bitmap_init makes a bitmap from RGB pixels, three bytes for each
// one.
int yzma_mtmd_bitmap_init(int nx, int ny, const unsigned char * data) {
    if (nx <= 0 || ny <= 0 || data == nullptr) {
        set_error("a bitmap needs a size and data: %d by %d", nx, ny);
        return YZMA_ERR_GENERIC;
    }

    mtmd_bitmap * bitmap = mtmd_bitmap_init((uint32_t) nx, (uint32_t) ny, data);
    if (bitmap == nullptr) {
        set_error("cannot make a bitmap of %d by %d", nx, ny);
        return YZMA_ERR_ALLOC;
    }
    return bitmaps.add(bitmap);
}

void yzma_mtmd_bitmap_free(int bitmap) {
    mtmd_bitmap * b = bitmaps.take(bitmap);
    if (b != nullptr) {
        mtmd_bitmap_free(b);
    }
}

int yzma_mtmd_input_chunks_init(void) {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        set_error("cannot make a list of chunks");
        return YZMA_ERR_ALLOC;
    }
    return chunk_lists.add(chunks);
}

void yzma_mtmd_input_chunks_free(int chunks) {
    mtmd_input_chunks * c = chunk_lists.take(chunks);
    if (c != nullptr) {
        mtmd_input_chunks_free(c);
    }
}

int yzma_mtmd_input_chunks_size(int chunks) {
    mtmd_input_chunks * c = chunk_lists.get(chunks);
    if (c == nullptr) {
        set_error("invalid chunks handle %d", chunks);
        return YZMA_ERR_HANDLE;
    }
    return (int) mtmd_input_chunks_size(c);
}

// yzma_mtmd_tokenize puts the text and the bitmaps into the list of chunks. The
// text must hold one marker for each bitmap. bitmap_handles points to n_bitmaps
// handles that came from yzma_mtmd_bitmap_init.
int yzma_mtmd_tokenize(int mctx, int chunks, const char * text, int text_len,
                       int add_special, int parse_special,
                       const int * bitmap_handles, int n_bitmaps) {
    mtmd_context * c = mtmd_contexts.get(mctx);
    mtmd_input_chunks * out = chunk_lists.get(chunks);
    if (c == nullptr || out == nullptr) {
        set_error("invalid handle: mtmd %d, chunks %d", mctx, chunks);
        return YZMA_ERR_HANDLE;
    }

    std::vector<const mtmd_bitmap *> media;
    media.reserve((size_t) (n_bitmaps > 0 ? n_bitmaps : 0));
    for (int i = 0; i < n_bitmaps; i++) {
        mtmd_bitmap * b = bitmaps.get(bitmap_handles[i]);
        if (b == nullptr) {
            set_error("invalid bitmap handle %d", bitmap_handles[i]);
            return YZMA_ERR_HANDLE;
        }
        media.push_back(b);
    }

    mtmd_input_text input = {};
    input.text          = text;
    input.text_len      = (size_t) text_len;
    input.add_special   = add_special != 0;
    input.parse_special = parse_special != 0;

    const int rc = mtmd_tokenize(c, out, &input, media.data(), media.size());
    if (rc != 0) {
        set_error("mtmd_tokenize returned %d", rc);
    }
    return rc;
}

// yzma_mtmd_helper_eval_chunks runs every chunk through the model: the text
// with llama_decode, and the media through the projector first. It returns the
// position after the last chunk.
int yzma_mtmd_helper_eval_chunks(int mctx, int ctx, int chunks, int n_past, int seq_id,
                                 int n_batch, int logits_last) {
    mtmd_context * c = mtmd_contexts.get(mctx);
    llama_context * lctx = contexts.get(ctx);
    mtmd_input_chunks * list = chunk_lists.get(chunks);
    if (c == nullptr || lctx == nullptr || list == nullptr) {
        set_error("invalid handle: mtmd %d, context %d, chunks %d", mctx, ctx, chunks);
        return YZMA_ERR_HANDLE;
    }

    llama_pos new_n_past = 0;
    const int rc = mtmd_helper_eval_chunks(c, lctx, list, (llama_pos) n_past,
                                           (llama_seq_id) seq_id, n_batch,
                                           logits_last != 0, &new_n_past);
    if (rc != 0) {
        set_error("mtmd_helper_eval_chunks returned %d", rc);
        return YZMA_ERR_GENERIC;
    }
    return (int) new_n_past;
}

//
// metadata
//

// copy_string copies s into buf and returns the number of bytes copied. A null
// s gives 0.
static int copy_string(const char * s, char * buf, int cap) {
    if (s == nullptr) {
        return 0;
    }
    const int n = (int) strlen(s);
    if (cap < n + 1) {
        return YZMA_ERR_TOO_SMALL;
    }
    memcpy(buf, s, (size_t) n + 1);
    return n;
}

int yzma_model_meta_count(int model) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_meta_count(m);
}

// The meta calls give the length of the whole value, as snprintf does. A length
// that is not less than cap tells that buf holds only the start of the value.
// A key that is not in the model gives -1.
int yzma_model_meta_key_by_index(int model, int i, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_meta_key_by_index(m, i, buf, (size_t) cap);
}

int yzma_model_meta_val_str_by_index(int model, int i, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_meta_val_str_by_index(m, i, buf, (size_t) cap);
}

int yzma_model_meta_val_str(int model, const char * key, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return llama_model_meta_val_str(m, key, buf, (size_t) cap);
}

// yzma_model_meta_key_str copies the name of a key of llama_model_meta_key into
// buf. A key that is not known gives 0.
int yzma_model_meta_key_str(int key, char * buf, int cap) {
    return copy_string(llama_model_meta_key_str((enum llama_model_meta_key) key), buf, cap);
}

// yzma_model_cls_label copies the label of output i of a classifier into buf. A
// model with no label for i gives 0.
int yzma_model_cls_label(int model, int i, char * buf, int cap) {
    llama_model * m = models.get(model);
    if (m == nullptr) {
        set_error("invalid model handle %d", model);
        return YZMA_ERR_HANDLE;
    }
    return copy_string(llama_model_cls_label(m, (uint32_t) i), buf, cap);
}

//
// state
//
// A state can be larger than an int, thus these give a double. A bad handle
// gives -1 and a failure of llama.cpp gives 0, as the C API does.
//

double yzma_state_get_size(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return -1;
    }
    return (double) llama_state_get_size(c);
}

double yzma_state_get_data(int ctx, uint8_t * dst, double size) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return -1;
    }
    return (double) llama_state_get_data(c, dst, (size_t) size);
}

double yzma_state_set_data(int ctx, const uint8_t * src, double size) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return -1;
    }
    return (double) llama_state_set_data(c, src, (size_t) size);
}

double yzma_state_seq_get_size_ext(int ctx, int seq_id, int flags) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return -1;
    }
    return (double) llama_state_seq_get_size_ext(c, seq_id, (llama_state_seq_flags) flags);
}

double yzma_state_seq_get_data_ext(int ctx, uint8_t * dst, double size, int seq_id, int flags) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return -1;
    }
    return (double) llama_state_seq_get_data_ext(c, dst, (size_t) size, seq_id, (llama_state_seq_flags) flags);
}

double yzma_state_seq_set_data_ext(int ctx, const uint8_t * src, double size, int dest_seq_id, int flags) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return -1;
    }
    return (double) llama_state_seq_set_data_ext(c, src, (size_t) size, dest_seq_id, (llama_state_seq_flags) flags);
}

//
// performance
//

// yzma_perf_context writes t_start_ms, t_load_ms, t_p_eval_ms, t_eval_ms,
// n_p_eval, n_eval, and n_reused into out, in that order, and returns 7. The
// times stay 0 for a context that has no_perf.
int yzma_perf_context(int ctx, double * out) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    const llama_perf_context_data data = llama_perf_context(c);
    out[0] = data.t_start_ms;
    out[1] = data.t_load_ms;
    out[2] = data.t_p_eval_ms;
    out[3] = data.t_eval_ms;
    out[4] = data.n_p_eval;
    out[5] = data.n_eval;
    out[6] = data.n_reused;
    return 7;
}

int yzma_perf_context_reset(int ctx) {
    llama_context * c = contexts.get(ctx);
    if (c == nullptr) {
        set_error("invalid context handle %d", ctx);
        return YZMA_ERR_HANDLE;
    }
    llama_perf_context_reset(c);
    return YZMA_OK;
}

// chain_of gives the sampler of a handle if it is a chain. llama.cpp stops the
// program when the perf calls get a sampler that is not a chain.
static llama_sampler * chain_of(int chain) {
    llama_sampler * s = samplers.get(chain);
    if (s == nullptr) {
        set_error("invalid sampler handle %d", chain);
        return nullptr;
    }
    if (strcmp(llama_sampler_name(s), "chain") != 0) {
        set_error("sampler %d is not a chain", chain);
        return nullptr;
    }
    return s;
}

// yzma_perf_sampler writes t_sample_ms and n_sample into out and returns 2.
int yzma_perf_sampler(int chain, double * out) {
    llama_sampler * s = chain_of(chain);
    if (s == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    const llama_perf_sampler_data data = llama_perf_sampler(s);
    out[0] = data.t_sample_ms;
    out[1] = data.n_sample;
    return 2;
}

int yzma_perf_sampler_reset(int chain) {
    llama_sampler * s = chain_of(chain);
    if (s == nullptr) {
        return YZMA_ERR_HANDLE;
    }
    llama_perf_sampler_reset(s);
    return YZMA_OK;
}

//
// system
//

int yzma_print_system_info(char * buf, int cap) {
    return copy_string(llama_print_system_info(), buf, cap);
}

int yzma_ftype_name(int ftype, char * buf, int cap) {
    return copy_string(llama_ftype_name((enum llama_ftype) ftype), buf, cap);
}

// A time in microseconds does not fit an int, thus this gives a double.
double yzma_time_us(void) {
    return (double) llama_time_us();
}

int yzma_max_devices(void) {
    return (int) llama_max_devices();
}

int yzma_max_parallel_sequences(void) {
    return (int) llama_max_parallel_sequences();
}

int yzma_supports_gpu_offload(void) {
    return (int) llama_supports_gpu_offload();
}

} // extern "C"
