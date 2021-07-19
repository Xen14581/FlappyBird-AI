import pygame
import neat
import os

from GameObjects.Bird import Bird
from GameObjects.Ground import Ground
from GameObjects.Pipes import Pipe
from GameObjects.Scoreboard import Scoreboard


def eval_genomes(genomes, config):
    SCREEN_SIZE = (450, 640)

    pygame.init()

    screen = pygame.display.set_mode(SCREEN_SIZE)

    networks = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(net)
        birds.append(Bird(SCREEN_SIZE[1] * 0.4))
        ge.append(genome)

    running = True

    clock = pygame.time.Clock()

    base = Ground(575)
    pipes = [Pipe(400)]
    scoreboard = Scoreboard()

    bg = pygame.image.load('GameObjects/sprites/bg.png').convert_alpha()

    while running and len(birds) > 0:
        clock.tick(120)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                break

        screen.blit(pygame.transform.scale(bg, SCREEN_SIZE), (0, 0))

        for pipe in pipes:
            pipe.draw(screen)
        base.draw(screen)
        scoreboard.show(screen, SCREEN_SIZE)
        for bird in birds:
            bird.draw(screen)

        next_pipe = 0
        if len(pipes) > 0 and birds[0].x > pipes[0].x + pipes[0].image.get_width():
            next_pipe = 1

        for bird in birds:
            index = birds.index(bird)
            ge[index].fitness += 0.1
            bird.move()

            output = networks[index].activate((bird.y, pipes[next_pipe].height, pipes[next_pipe].bottom))

            if output[0] > 0.5:
                bird.jump()

        base.move()

        if pipes[-1].x <= 200:
            pipes.append(Pipe(500))
        if pipes[0].x == 50:
            scoreboard.increment()

        for pipe in pipes:
            pipe.move()

            if pipe.x <= -100:
                pipes.remove(pipe)

            for bird in birds:
                if pipe.collide(bird) or base.collide(bird) or bird.y <= -50:
                    ge[birds.index(bird)].fitness -= 0.5
                    networks.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

        pygame.display.update()


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 30)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
