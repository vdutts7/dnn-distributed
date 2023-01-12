
<div align="center">

<img src="https://res.cloudinary.com/dnz16usmk/image/upload/v1708598121/dnn.png" alt="logo" width="100" height="80"  />

  <h1 align="center">
        Distributed DNNs (Deep Neural Networks)
    </h1>
    <p align="center"> 
        <i><b>C++/MPI proxies to perform distributed training of DNNs</b></i>
        <br /> 
    </p>

[![Github][github]][github-url]


 </div>

<br/>

## Table of Contents

  <ol>
    <a href="#about">📝 About</a><br/>
    <a href="#how-to-build">💻 How to build</a><br/>
    <a href="#tools-used">🔧 Tools used</a>
        <ul>
        </ul>
    <a href="#contact">👤 Contact</a>
  </ol>

<br/>

## 📝About

C++/MPI proxies to perform distributed training of DNNs (deep neural networks):
- `GPT-2`
- `GPT-3`
- `CosmoFlow`
- `DLRM`

These proxies cover:
- *Data parallelism*: same NN replicated across multiple processors, but each copy processes a different subset of the data
- *Operator parallelism*: splitting different operations (i.e. layers) of a NN across multiple processors
- *Pipeline parallelism*: different stages of a NN are processed on different processors, in a pipelined fashion
- *Hybrid parallelism*: combines two or more of the above types of parallelism i.e. different parts of the NN are processed in parallel across different processors AND data is also split across processors

### Benchmarking GPU interconnect performance • NCCL/MPI

- **MPI for distributed training**: managing communication between nodes in a distributed system, enabling efficient data parallelism and model parallelism strategies
- **NCCL for optimized GPU communication**: common communication operations such as `all-reduce` performed on NVIDIA GPUs


## Scaling techniques for model parallelism

- **Essential for large model** training i.e. ones that don't even fit into the memory of a single GPU
- **The GPT-3 example** shows a hybrid approach to model and data parallelism. Scaling out training of extremely large models (GPT-3 has over >150 billion paramaters) across multiple GPUs and nodes


## Optimizing CNNs

- **The CosmoFlow example** illustrates distributed training of a CNN, leveraging GPU acceleration for performance gains.



## 💻 How to build

Compile via:

`mpicxx communications/gpt-2.cpp -o gpt-2`

Then run:

`mpirun -n 32 ./gpt-2`

Set the total num of **Transformer layers** AND total num of **pipeline stages**:

`mpirun -n 32 ./gpt-2 64 8`



## 🔧Tools Used

<img
  src="https://img.shields.io/badge/C++-4F4F4F?style=for-the-badge&logo=cplusplus&color=navy"
  alt="C++"
/>
<img
  src="https://img.shields.io/badge/MPI (Message Passing Interface)-40B5A4?style=for-the-badge&color=black"
  alt="MPI"
/>
<img
src="https://img.shields.io/badge/NCCL_(NVIDIA Collective Communications Library)-40B5A4?style=for-the-badge&logo=nvidia&logoColor=ffffff&color=76b900"
alt="NCCL"
/>
<img
src="https://img.shields.io/badge/pyTorch-EE4C2C?style=for-the-badge&logo=pyTorch&logoColor=white&color=EE4C2C"
alt="pyTorch"
/>

## 👤Contact

<!-- Replace placeholders with your actual contact information -->
[![Email][email]][email-url]
[![Twitter][twitter]][twitter-url]

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[email]: https://img.shields.io/badge/me@vd7.io-FFCA28?style=for-the-badge&logo=Gmail&logoColor=00bbff&color=black
[email-url]: #
[github]: https://img.shields.io/badge/Github-2496ED?style=for-the-badge&logo=github&logoColor=white&color=black
[github-url]: https://github.com/vdutts7/dnn-distributed
[twitter]: https://img.shields.io/badge/Twitter-FFCA28?style=for-the-badge&logo=Twitter&logoColor=00bbff&color=black
[twitter-url]: https://twitter.com/vdutts7/