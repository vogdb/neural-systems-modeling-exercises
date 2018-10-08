import numpy as np


def to_phenotype(population):
    '''
    converts chromosomes to the decimal numbers they represent
    '''
    exp_of_2 = np.exp2(range(population.shape[1] - 1, -1, -1))
    x = population.dot(exp_of_2)
    return x ** 2 - 30 * x + 230


population_size = 5
chromosome_len = 5
generation_num = 100
mutation_rate = 0.02
# each row is a genotype, e.g. chromosome, each column is a gene
population = np.random.randint(0, 2, (population_size, chromosome_len))
generation_mean_record = np.zeros(generation_num)
for g in range(generation_num):
    phenotype = to_phenotype(population)
    generation_mean_record[g] = np.mean(phenotype)
    if g == generation_num - 1:
        break
    phenotype_norm = phenotype / np.sum(phenotype)
    rand_delta = np.random.rand(len(phenotype))
    rand_delta_norm = rand_delta / np.sum(rand_delta)
    phenotype_norm_pert = phenotype_norm + rand_delta_norm
    phenotype_norm_pert_sort_idx = np.argsort(phenotype_norm_pert)
    dad = population[phenotype_norm_pert_sort_idx[0]]
    mom = population[phenotype_norm_pert_sort_idx[1]]
    crs_over = np.random.randint(1, chromosome_len)
    son = np.concatenate((dad[:crs_over], mom[crs_over:]))
    daughter = np.concatenate((mom[:crs_over], dad[crs_over:]))
    population[phenotype_norm_pert_sort_idx[-1]] = son
    population[phenotype_norm_pert_sort_idx[-2]] = daughter
    mutation = np.random.random(population.shape) < mutation_rate
    population = np.abs(population - mutation)

print(generation_mean_record)
