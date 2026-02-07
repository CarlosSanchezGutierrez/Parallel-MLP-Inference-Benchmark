#include <bits/stdc++.h>
#include <thread>
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
    uint32_t m = x & 0x00FFFFFFU;          // 24-bit fraction
    float f = (float)m / (float)0x00800000U; // ~[0,2)
    return f - 1.0f;                       // ~[-1,1)
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
static inline float relu(float x){ return x > 0.f ? x : 0.f; }

int main(int argc, char** argv){
    if(argc < 6){
        cerr << "Usage: ./mlp_thread B D H C P\n";
        return 1;
    }
    int B = stoi(argv[1]);
    int D = stoi(argv[2]);
    int H = stoi(argv[3]);
    int C = stoi(argv[4]);
    int P = stoi(argv[5]);
    if(B<=0||D<=0||H<=0||C<=0||P<=0){
        cerr << "All params must be > 0\n";
        return 1;
    }

    // Inputs
    vector<float> X((size_t)B * D);
    for(int b=0;b<B;b++)
        for(int d=0;d<D;d++)
            X[(size_t)b*D + d] = gen_input(b,d);

    // Transposed weights for better streaming:
    // W1T[d*H + h] = W1[h*D + d]
    vector<float> W1T((size_t)D * H);
    vector<float> b1((size_t)H);
    for(int h=0; h<H; h++){
        b1[h] = gen_b1(h);
        for(int d=0; d<D; d++){
            W1T[(size_t)d*H + h] = gen_w1(h,d);
        }
    }

    // W2T[h*C + c] = W2[c*H + h]
    vector<float> W2T((size_t)H * C);
    vector<float> b2((size_t)C);
    for(int c=0; c<C; c++){
        b2[c] = gen_b2(c);
        for(int h=0; h<H; h++){
            W2T[(size_t)h*C + c] = gen_w2(c,h);
        }
    }

    auto t0 = chrono::high_resolution_clock::now();

    vector<double> partial((size_t)P, 0.0);

    auto worker = [&](int tid){
        int b0 = (int)((long long)B * tid / P);
        int b1i= (int)((long long)B * (tid+1) / P);

        vector<float> hidden((size_t)H);
        vector<float> out((size_t)C);
        double local = 0.0;

        for(int b=b0; b<b1i; b++){
            const float* x = &X[(size_t)b*D];

            // hidden = b1
            memcpy(hidden.data(), b1.data(), (size_t)H*sizeof(float));

            // hidden += sum_d x[d] * W1T[d,:]
            for(int d=0; d<D; d++){
                float xd = x[d];
                const float* wrow = &W1T[(size_t)d*H];
                for(int h=0; h<H; h++){
                    hidden[h] += xd * wrow[h];
                }
            }

            // ReLU in-place
            for(int h=0; h<H; h++) hidden[h] = relu(hidden[h]);

            // out = b2
            memcpy(out.data(), b2.data(), (size_t)C*sizeof(float));

            // out += sum_h hidden[h] * W2T[h,:]
            for(int h=0; h<H; h++){
                float hv = hidden[h];
                const float* wrow = &W2T[(size_t)h*C];
                for(int c=0; c<C; c++){
                    out[c] += hv * wrow[c];
                }
            }

            // checksum
            for(int c=0; c<C; c++) local += (double)out[c];
        }

        partial[(size_t)tid] = local;
    };

    vector<thread> th;
    th.reserve((size_t)P);
    for(int t=0;t<P;t++) th.emplace_back(worker,t);
    for(auto& tt: th) tt.join();

    double checksum = 0.0;
    for(double v: partial) checksum += v;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    cout.setf(ios::fixed);
    cout << "runtime_ms " << setprecision(3) << ms << "\n";
    cout << "checksum " << setprecision(6) << checksum << "\n";
    return 0;
}
