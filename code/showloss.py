import matplotlib.pyplot as plt

f = open('save_loss.txt', 'r')
a = [i.split() for i in f.readlines()]
f.close()

x = [i[0] for i in a]
y = [i[1] for i in a]

x = [int(i[5:-1]) for i in x]
y = [float(i[9:]) for i in y]

plt.plot(x, y)
plt.show()

