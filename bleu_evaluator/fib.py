import sys


def fib(n: int) -> int:
    prev, cur = 0, 1
    for _ in range(n - 1):
        prev, cur = cur, prev + cur
    return cur


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("This program must be executed with one argument!")
        exit(1)

    try:
        num = int(sys.argv[1])
    except ValueError:
        print("Argument must be an integer!")
        exit(1)

    if num <= 0:
        print("Argument must be positive!")
        exit(1)

    print(fib(num))
