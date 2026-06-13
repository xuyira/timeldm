## TimeLDM: Latent Diffusion Model for Unconditional Time Series Generation
### Abstract
In this paper, we propose TimeLDM, a novel latent diffusion model for high-quality time series generation. TimeLDM is composed of a variational autoencoder that encodes time series into an informative and smoothed latent content and a latent diffusion model operating in the latent space to generate latent information. We evaluate the ability of our method to generate synthetic time series with simulated and real-world datasets and benchmark the performance against existing state-of-the-art methods. Qualitatively and quantitatively, we find that the proposed TimeLDM persistently delivers high-quality generated time series. For example, TimeLDM achieves new state-of-the-art results on the simulated benchmarks and an average improvement of 55% in Discriminative score with all benchmarks. Further studies demonstrate that our method yields more robust outcomes across various lengths of time series data generation. Especially, for the Context-FID score and Discriminative score, TimeLDM realizes significant improvements of 80% and 50%, respectively.

### Pipeline
![](./figure/timeldm.png)



