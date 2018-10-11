import numpy as np


def laminate_array_to_matrix(array):
    n = len(array)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, :] = np.roll(array, i)
    return matrix


def bubble_network(narrow_sd, wide_sd, wide_w):
    n_units = 51
    interval = np.arange(-int(n_units / 2), -int(n_units / 2) + n_units)
    narrow = gauss_pro(interval, narrow_sd)
    wide = wide_w * gauss_pro(interval, wide_sd)
    diff_of_gauss = narrow - wide
    diff_of_gauss = np.roll(diff_of_gauss, -int(n_units / 2))
    W = laminate_array_to_matrix(diff_of_gauss)
    signal = np.zeros(n_units)
    signal[24:27] = 1
    noise = np.random.rand(n_units)
    x = signal + noise

    y = np.zeros(n_units)
    for t_i in range(1, 20):
        y = W.dot(y) + np.eye(n_units).dot(x)
        y = np.clip(y, 0, 10)

    return np.sum(np.concatenate((20 * (10 - y[24:27]), y[:24], y[27:])))


def gauss_pro(s, sd):
    return np.exp((s / sd) ** 2 * (-0.5))


def decode_population(population):
    '''
    converts chromosomes to probabilities they represent
    '''
    exp_of_2 = np.exp2(range(4, -1, -1))
    narrow_sd_list = population[:, :4].dot(exp_of_2[1:])
    narrow_sd_list[narrow_sd_list == 0] = 0.001
    wide_sd_list = population[:, 4:9].dot(exp_of_2)
    wide_sd_list[wide_sd_list == 0] = 0.001
    wide_w_list = population[:, 9:].dot(exp_of_2)
    wide_w_list = 0.1 + 0.013 * wide_w_list
    return narrow_sd_list, wide_sd_list, wide_w_list


def fitness_idx(phenotype):
    phenotype_norm = phenotype / np.sum(phenotype)
    rand_delta = np.random.rand(len(phenotype))
    rand_delta_norm = rand_delta / np.sum(rand_delta)
    phenotype_norm_pert = phenotype_norm + rand_delta_norm / 3
    return np.argsort(phenotype_norm_pert)


def offspring(population, phenotype_fit_idx):
    family_offspring(population, phenotype_fit_idx)
    family_offspring(population, phenotype_fit_idx[2:-2])
    return population


def family_offspring(population, phenotype_fit_idx):
    dad = population[phenotype_fit_idx[0]]
    mom = population[phenotype_fit_idx[1]]
    crs_over = np.random.randint(1, chromosome_len)
    son = np.concatenate((dad[:crs_over], mom[crs_over:]))
    daughter = np.concatenate((mom[:crs_over], dad[crs_over:]))
    population[phenotype_fit_idx[-1]] = son
    population[phenotype_fit_idx[-2]] = daughter
    return population


def mutate(population):
    mutation = np.random.random(population.shape) < mutation_rate
    return np.abs(population - mutation)


population_size = 15
chromosome_len = 5 + 4 + 5  # 5 - wide gauss, 4 - narrow gauss, 5 - wide gauss weight
generation_num = 30
mutation_rate = 0.01
# each row is a genotype, e.g. chromosome, each column is a bit of one of three genes
population = np.random.randint(0, 2, (population_size, chromosome_len))
previous_prb_list = np.zeros(population_size)
diff_list = np.zeros(population_size)
generation_diff_record = np.zeros(generation_num)

for g in range(generation_num):
    narrow_sd_list, wide_sd_list, wide_w_list = decode_population(population)
    for i_chrom in range(population_size):
        narrow_sd = narrow_sd_list[i_chrom]
        wide_sd = wide_sd_list[i_chrom]
        wide_w = wide_w_list[i_chrom]
        diff_list[i_chrom] = bubble_network(narrow_sd, wide_sd, wide_w)

    generation_diff_record[g] = np.mean(diff_list)
    if g == generation_num - 1:
        break

    phenotype_fit_idx = fitness_idx(diff_list)
    population = offspring(population, phenotype_fit_idx)
    population = mutate(population)

print('Mean diff:\n{}'.format(generation_diff_record))
