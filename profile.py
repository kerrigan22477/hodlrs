import cProfile
from main import Main

with cProfile.Profile() as pr:
    for i in range(10):
        print(i)

pr.print_stats()