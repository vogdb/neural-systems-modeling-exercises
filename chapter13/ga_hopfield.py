import numpy as np


def hopfield_process(est_prb):
    n_units = 20
    n_patterns = 3
    actual_prb = .5
    # number of async updates to train hopfield network
    n_its = 200
    # auto-connector mask that nullifies self connections
    msk = np.ones(n_units) - np.eye(n_units)
    patterns = np.random.rand(n_patterns, n_units) < actual_prb
    # stable states of patterns reached during the auto-connector network
    pattern_stable_state = np.zeros((n_patterns, n_units))

    hopfield_net = (patterns.T - est_prb).dot(patterns - est_prb)
    hopfield_net = hopfield_net * msk

    for p in range(n_patterns):
        # start with pattern of 0.5 strength
        y = patterns[p, :] * .5
        for i in range(n_its):
            rnd_unit_idx = np.random.randint(0, n_units)
            q = hopfield_net[rnd_unit_idx, :].dot(y)
            y = q > 0
        pattern_stable_state[p, :] = y

    return np.sum(np.abs(patterns - pattern_stable_state))


def decode_population(population):
    '''
    converts chromosomes to probabilities they represent
    '''
    exp_of_2 = np.exp2(range(population.shape[1] - 1, -1, -1))
    return 0.01 + (0.49 / 31) * population.dot(exp_of_2)


def fitness_idx(phenotype):
    if np.sum(phenotype) == 0:
        phenotype_norm = phenotype
    else:
        phenotype_norm = phenotype / np.sum(phenotype)
    rand_delta = np.random.rand(len(phenotype))
    rand_delta_norm = rand_delta / np.sum(rand_delta)
    phenotype_norm_pert = phenotype_norm + rand_delta_norm / 3
    return np.argsort(phenotype_norm_pert)


def offspring(population, phenotype_fit_idx):
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


population_size = 5
chromosome_len = 5
generation_num = 30
mutation_rate = 0.01
# each row is a genotype, e.g. chromosome, each column is bit of a single gene
population = np.random.randint(0, 2, (population_size, chromosome_len))
previous_prb_list = np.zeros(population_size)
diff_list = np.zeros(population_size)
generation_diff_record = np.zeros(generation_num)
generation_prb_est_record = np.zeros(generation_num)

for g in range(generation_num):
    est_prb_list = decode_population(population)
    for i_chrom in range(population_size):
        est_prb = est_prb_list[i_chrom]
        prev_est_prb = previous_prb_list[i_chrom]
        if est_prb - prev_est_prb > 1e-5:
            diff_list[i_chrom] = hopfield_process(est_prb)

    generation_diff_record[g] = np.mean(diff_list)
    generation_prb_est_record[g] = np.mean(est_prb_list)
    if g == generation_num - 1:
        break

    phenotype_fit_idx = fitness_idx(diff_list)
    population = offspring(population, phenotype_fit_idx)
    population = mutate(population)

print('Mean prb estimation, the actual is 0.5,:\n{}'.format(generation_prb_est_record))
print('Mean diff:\n{}'.format(generation_diff_record))
