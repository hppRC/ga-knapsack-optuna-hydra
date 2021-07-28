from typing import Callable, List, Optional
import hydra
from omegaconf import DictConfig
import random
from dataclasses import dataclass


@dataclass
class KnapsackProblem:
    T: int
    N: int
    L: int
    mut: float
    cross: float
    size: List[int]
    importance: List[int]
    weight_limit: int

    @staticmethod
    def gtypeToPtype(gtype):
        return gtype[:]

    @staticmethod
    def selectAnAgentByRoulette(pop):
        total = sum(i.fitness for i in pop)
        val = random.random() * total
        for i in pop:
            val -= i.fitness
            if (val <= 0):
                return i

    def fitnessFunction(self, p):
        weightSum, impSum = 0, 0
        for i, g in enumerate(p):
            if g:
                weightSum += self.size[i]
                impSum += self.importance[i]
        if weightSum <= self.weight_limit:
            return impSum
        else:
            return 0

    def crossover(self, a1, a2):
        point = random.randint(1, self.L-1)
        for i in range(point, self.L):
            a1.genotype[i], a2.genotype[i] = a2.genotype[i], a1.genotype[i]


@dataclass
class Agent:
    problem: KnapsackProblem
    genotype: List[int]
    phenotype: Optional[List[int]] = None
    fitness: float = 0.0

    def getOffspring(self):
        o = Agent(problem = self.problem, genotype = self.genotype)
        for i in range(self.problem.L):
            if random.random() < self.problem.mut:
                o.genotype[i] = 1 - o.genotype[i]
        return(o)

    def develop(self, dfunc: Callable):
        self.phenotype = dfunc(self.genotype)

    def evaluate(self, efunc: Callable):
        self.fitness = efunc(self.phenotype)

    def __repr__(self):
        genotype_str = ''.join(str(x) for x in self.genotype)
        phenotype_str = ''.join(str(x) for x in self.phenotype)

        return f"genotype:\t{genotype_str}\nphenotype:\t{phenotype_str}\nfitness:\t{self.fitness}\n"


def solve(problem: KnapsackProblem) -> Agent:
    population = []
    for _ in range(problem.N):
        genotype = [random.randint(0, 1) for _ in range(problem.L)]
        agent = Agent(problem = problem, genotype = genotype)
        population.append(agent)
    best = population[0]

    for _ in range(problem.T):
        for a in population:
            a.develop(problem.gtypeToPtype)
            a.evaluate(problem.fitnessFunction)
            if(a.fitness>best.fitness):
                best= a

        newpop = []
        for _ in range(problem.N // 2):
            n1 = problem.selectAnAgentByRoulette(population).getOffspring()
            n2 = problem.selectAnAgentByRoulette(population).getOffspring()

            if random.random() < problem.cross:
                problem.crossover(n1, n2)
            newpop.append(n1)
            newpop.append(n2)

        population = newpop
    return best


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config : DictConfig) -> None:
    problem = KnapsackProblem(
        T = config.T,
        N = config.N,
        L = config.L,
        mut = config.mut,
        cross = config.cross,
        size = config.size,
        importance = config.importance,
        weight_limit = config.weight_limit,
    )
    sum = 0
    for _ in range(config.times):
        best = solve(problem)
        sum += best.fitness
    avg = sum / config.times
    print(avg)
    return avg


if __name__ == "__main__":
    main()