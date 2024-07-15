# SDPR_admix
SDPRX_admix is a statistical method for cross-population prediction of complex traits. It integrates GWAS summary statistics and LD matrices from one populations (EUR and non-EUR) and individual data with ancestry of ERU and non_EUR to compuate polygenic risk scores.

# Installation
You can download SDPR_admix by simply running
```bash
git clone https://github.com/FreyaWa26/SDPR_admix.git
```
SDPR_admix is developed under python 3. We recommend you to run SDPR_admix in the Anaconda so that libraries like numpy and scipy would be pre-installed. If running in the Anaconda environment, the only requirement to run SDPR_admix would be installing joblib.

# Input 
## Individual data
Local ancestry files with genotype of 0,1,2. The file should have no header. Coulmns are individuals and rows are SNPs.<br>
E.G. X1.shape = (15357, 10000)


## Reference LD
The reference LD matrices based on 1000 Genome Hapmap3 SNPs can be downloaded from the following link.

| Populations | Number of SNPs | Size | Link |
|-------------|----------------|------|------|
| EUR_EAS     | 873,166        | 6.3G | [link](https://drive.google.com/file/d/1MGt-Ai5foThXBF1xAZMKksBRqZGsbQ1l/view) |
| EUR_AFR     | 903,499        | 8.7G | [link](https://drive.google.com/file/d/1cbcfCicsuARfcv231tY98PTnAoOoQS8O/view) |

## Summary Statistics
The summary statistics should have at least following columns with the same name, where SNP is the marker name, A1 is the effect allele, A2 is the alternative allele, Z is the Z score for the association statistics, and N is the sample size.
```
SNP     A1      A2      Z       N
rs737657        A       G       -2.044      252156
rs7086391       T       C       -2.257      248425
rs1983865       T       C       3.652    253135
...
```
## Running SDPR_admix
```bash
python SimulationAPP.py --ss1 X_afr.txt --ss2 X_eur.txt --load_ld LD.gz --N3 10000 --N 10000 --n_sim 1000 --burn 500 --p_sparse 0.9 --out /test/ --rho 0.9 0.7 0.8 --ID 1
```

The output file has the following format:

| beta1              | beta2              | beta3              | beta10 | beta20 | beta30 | beta3_hat              |
|--------------------|--------------------|--------------------|--------|--------|--------|------------------------|
| -0.000251 | -0.0000903  | 0.0000639 | 0.0    | 0.0    | 0.0    | 0.00389  |
| -0.000633 | -0.0005852  | -0.000250 | 0.0    | 0.0    | 0.0    | 0.00120  |


