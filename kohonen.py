import numpy as np
import pygame
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_input, pixels):
        self.w = pixels * np.random.rand(n_input)

class Network:
    def __init__(self, n_neurons, n_input, pixels, epsilon, sigma, eps_decay, sigma_decay):
        self.pixels = pixels
        self.epsilon = epsilon
        self.sigma = sigma
        self.eps_decay = eps_decay
        self.sigma_decay = sigma_decay
        self.n_neurons = n_neurons
        self.neurons = np.array([])
        self.i_best = 0
        for i in range(n_neurons):
            self.neurons = np.append(self.neurons, Neuron(n_input, pixels))

    def learn(self, example):

        # find winning neuron
        self.i_best = 0
        dist_best = float("inf")
        for i in range(self.n_neurons):
            neuron = self.neurons[i]
            dist = np.dot(neuron.w - example, neuron.w - example)
            if dist <= dist_best:
                dist_best = dist
                self.i_best = i

        # calculate weight update
        for i in range(self.n_neurons):
            self.neurons[i].w += self.epsilon * f(i, self.i_best, self.sigma) * (example - self.neurons[i].w)

        # update parameters
        self.epsilon = self.eps_decay * self.epsilon
        self.sigma   = self.sigma_decay * self.sigma

    # calculate error
    def error(self, examples):
        E = 0
        for example in examples:

            # find winning neuron for each example
            self.i_best = 0
            dist_best = float("inf")
            for i in range(self.n_neurons):
                neuron = self.neurons[i]
                dist = np.dot(neuron.w - example, neuron.w - example)
                if dist <= dist_best:
                    dist_best = dist
                    self.i_best = i

            # update E
            for i in range(self.n_neurons):
                E += 0.5 * f(i, self.i_best, self.sigma) * np.dot(example - self.neurons[i].w, example - self.neurons[i].w)
       
        return E


# calculate f
def f(i, i_best, sigma):
    return np.exp(-(i - i_best)**2/(2 * sigma**2))

# draw neurons
def draw(network, pygame, screen):
    for i in range(network.n_neurons):

        pygame.draw.circle(screen, (200, 200, 200),
                          (network.neurons[i].w[0], network.neurons[i].w[1]),
                           5, # radius
                           )
        if i >= 1: # (there is no neuron before the first)
            pygame.draw.line(screen, (200, 200, 200),
                            (network.neurons[i].w[0], network.neurons[i].w[1]),
                            (network.neurons[i-1].w[0], network.neurons[i-1].w[1])
                            )

# screen setup
pixels = 1000
pygame.init()
screen = pygame.display.set_mode([pixels, pixels])

# initialize network
net = Network(
    n_neurons = 400,
    n_input = 2,
    pixels = pixels,
    epsilon = 0.30,
    sigma = 100,
    eps_decay = 0.999,
    sigma_decay = 0.995 #0.96
)

### different types of examples ###
def generate_example1(m, pixels):
    example = []
    for i in range(m):
        x = np.random.randint(50, pixels-50)
        y = (pixels/float(6)) * np.sin(8 * x/float(pixels)) + pixels/2.
        example.append([x, y])
    return example

def generate_example2(m, pixels):
    example = []
    for i in range(m):
        x = np.random.randint(50, pixels-50)
        y = np.random.randint(50, pixels-50)
        example.append([x, y])
    return example

def generate_example3(m, pixels):
    example = []
    for i in range(int(m/2)):
        x = np.random.randint(50, pixels-50)
        y = np.random.randint(50, pixels-50)
        example.append([x, y])
    for i in range(int(m/2)):
        x = np.random.randint(225, pixels-225)
        y = np.random.randint(225, pixels-225)
        example.append([x, y])
    return example

def generate_example4(m, pixels):
    example = []
    scale = 100
    for i in range(m):
        x = pixels/2 + pixels/3 * np.cos(2*np.pi * i/scale)
        y = pixels/2 + pixels/3 * np.sin(2*np.pi * i/scale)
        example.append([x, y])
    return example

### ###

# Choose example type from input
example_type = input("Enter example type (1, 2, 3 or 4): ")

m = 2500
if example_type == "1":
    examples = np.array(generate_example1(m, pixels))
elif example_type == "2":
    examples = np.array(generate_example2(m, pixels))
elif example_type == "3":  
    examples = np.array(generate_example3(m, pixels))
elif example_type == "4":
    examples = np.array(generate_example4(m, pixels))

# draw examples
def draw_examples(examples):
    for i in range(examples.shape[0]):
        pygame.draw.circle(screen, (100, 200, 100),
                          (examples[i][0], examples[i][1]),
                           5, # radius
                           )


err = 0
err_min = 1e3
errors = []
running = True
t = 0
t_max = 2e3
t_step = 50

while running:

    # stop conditions
    if (t >= t_max) or (net.sigma == 0): # (t_max is equivalent to eps_min)
        break

    # screen setup
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))

    #### 

    # draw examples
    draw_examples(examples)

    # pick random example
    i_example = np.random.randint(0, len(examples))
    example = examples[i_example]

    # learn
    net.learn(example)

    # uncomment to calculate and display error
    """
    if t % t_step == 0:
        err = net.error(examples)
        errors.append(err)
        plt.clf()
        plt.yscale('log',base=10) 
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Error', fontsize=20)
        plt.scatter([t_step * i for i in range(len(errors))], errors)
        plt.pause(0.001)
        
        if err <= err_min:
            break
    """

    # draw neurons
    draw(net, pygame, screen)

    pygame.display.flip()

    t+=1


plt.show()
pygame.quit()
