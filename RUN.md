# Cómo reproducir

## CPU (Codespace)
g++ -O3 -std=c++17 mlp_thread.cpp -o mlp_thread -pthread
g++ -O3 -std=c++17 -fopenmp mlp_openmp.cpp -o mlp_openmp
g++ -O3 -std=c++17 -mavx2 -mfma mlp_avx2.cpp -o mlp_avx2

Logs:
- log_2a.txt (correctness)
- log_2b.txt (threads vs time)
- log_2c.txt (batch vs time)
- log_2d.txt (AVX2 vs OpenMP)

## GPU (Colab)
Se corrió CUDA-only en Tesla T4. Resultados en log_2e_gpu.txt y reporte_colab.pdf
