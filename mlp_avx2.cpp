#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;

static inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static inline float u32_to_float_signed(uint32_t x) {
    uint32_t m = x & 0x00FFFFFFU;
    float f = (float)m / (float)0x00800000U;
    return f - 1.0f;
}
static inline float gen_input(int b, int d) {
    uint32_t idx = (uint32_t)(b * 1315423911u) ^ (uint32_t)(d * 2654435761u) ^ 0xA5A5A5A5u;
    return 0.5f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_w1(int h, int d) {
    uint32_t idx = (uint32_t)(h * 2246822519u) ^ (uint32_t)(d * 3266489917u) ^ 0x12345678u;
    return 0.1f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_b1(int h) {
    uint32_t idx = (uint32_t)(h * 374761393u) ^ 0x0BADF00Du;
    return 0.01f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_w2(int c, int h) {
    uint32_t idx = (uint32_t)(c * 1103515245u) ^ (uint32_t)(h * 12345u) ^ 0x87654321u;
    return 0.1f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_b2(int c) {
    uint32_t idx = (uint32_t)(c * 2654435761u) ^ 0xCAFEBABEu;
    return 0.01f * u32_to_float_signed(hash_u32(idx));
}

static inline void relu_inplace(float* a, int n){
    for(int i=0;i<n;i++) a[i] = (a[i] > 0.f) ? a[i] : 0.f;
}

static inline float hsum256_ps(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

int main(int argc, char** argv){
    if(argc < 5){
        cerr << "Usage: ./mlp_avx2 B D H C\n";
        return 1;
    }
    int B = stoi(argv[1]);
    int D = stoi(argv[2]);
    int H = stoi(argv[3]);
    int C = stoi(argv[4]);
    if(B<=0||D<=0||H<=0||C<=0){
        cerr << "All params must be > 0\n";
        return 1;
    }

    vector<float> X((size_t)B * D);
    for(int b=0;b<B;b++)
        for(int d=0;d<D;d++)
            X[(size_t)b*D + d] = gen_input(b,d);

    vector<float> W1T((size_t)D * H);
    vector<float> b1((size_t)H);
    for(int h=0; h<H; h++){
        b1[h] = gen_b1(h);
        for(int d=0; d<D; d++){
            W1T[(size_t)d*H + h] = gen_w1(h,d);
        }
    }

    vector<float> W2T((size_t)H * C);
    vector<float> b2((size_t)C);
    for(int c=0; c<C; c++){
        b2[c] = gen_b2(c);
        for(int h=0; h<H; h++){
            W2T[(size_t)h*C + c] = gen_w2(c,h);
        }
    }

    vector<float> hidden((size_t)H);
    vector<float> out((size_t)C);

    auto t0 = chrono::high_resolution_clock::now();

    double checksum = 0.0;

    for(int b=0; b<B; b++){
        const float* x = &X[(size_t)b*D];

        memcpy(hidden.data(), b1.data(), (size_t)H*sizeof(float));

        // hidden += sum_d x[d] * W1T[d,:]  (vectorize over H)
        for(int d=0; d<D; d++){
            __m256 xv = _mm256_set1_ps(x[d]);
            const float* wrow = &W1T[(size_t)d*H];

            int h = 0;
            for(; h + 8 <= H; h += 8){
                __m256 hv = _mm256_loadu_ps(hidden.data() + h);
                __m256 wv = _mm256_loadu_ps(wrow + h);
                hv = _mm256_fmadd_ps(xv, wv, hv);
                _mm256_storeu_ps(hidden.data() + h, hv);
            }
            for(; h < H; h++){
                hidden[h] += x[d] * wrow[h];
            }
        }

        relu_inplace(hidden.data(), H);

        memcpy(out.data(), b2.data(), (size_t)C*sizeof(float));

        // out += sum_h hidden[h] * W2T[h,:] (vectorize over C)
        for(int h=0; h<H; h++){
            __m256 hv = _mm256_set1_ps(hidden[h]);
            const float* wrow = &W2T[(size_t)h*C];

            int c = 0;
            for(; c + 8 <= C; c += 8){
                __m256 ov = _mm256_loadu_ps(out.data() + c);
                __m256 wv = _mm256_loadu_ps(wrow + c);
                ov = _mm256_fmadd_ps(hv, wv, ov);
                _mm256_storeu_ps(out.data() + c, ov);
            }
            for(; c < C; c++){
                out[c] += hidden[h] * wrow[c];
            }
        }

        // checksum
        // vectorize sum a bit
        int c = 0;
        __m256 sumv = _mm256_setzero_ps();
        for(; c + 8 <= C; c += 8){
            __m256 ov = _mm256_loadu_ps(out.data() + c);
            sumv = _mm256_add_ps(sumv, ov);
        }
        double local = (double)hsum256_ps(sumv);
        for(; c < C; c++) local += (double)out[c];
        checksum += local;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    cout.setf(ios::fixed);
    cout << "runtime_ms " << setprecision(3) << ms << "\n";
    cout << "checksum " << setprecision(6) << checksum << "\n";
    return 0;
}
