# CudaRuntime1 – CUDA matmul öğrenme projesi

## Derleme ve çalıştırma
- scripts\test.bat (derle + test), scripts\profile.bat [simt|wmma|both]
- Elle (CudaRuntime1/ klasöründen): nvcc -O3 -arch=sm_89 -lineinfo -Isrc/core -Isrc/reference -Isrc/simt -Isrc/wmma -Itests tests/main.cu tests/tests.cu src/reference/matmul.cu src/wmma/matmul_wmma.cu src/core/utils.cpp -o test.exe
- GPU: RTX 4070 Laptop (sm_89). Benchmark için Release kullan, Debug (-G) değil.
- Profil: scripts\profile.bat (ncu yönetici yetkisi ister), raporlar profiles/ içinde.

## Kod stili
- Girinti 4 boşluk, tab yok.

## Çalışma şekli
- Bu bir öğrenme reposu: ben istemeden kod değiştirme, önce kontrol edip söyle.
- Commit'e sadece kaynak dosyaları gir (exe ve .ncu-repz hariç).
