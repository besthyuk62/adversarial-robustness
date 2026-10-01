attacks = ["FGSM", "PGD", "CW"]
epsilons = [0.01, 0.03, 0.05]
def fff(a, b):
    print("Attack:", a, "/ Epsilon:", b)
for attack, epsilon in zip(attacks, epsilons):
    fff(attack, epsilon)


