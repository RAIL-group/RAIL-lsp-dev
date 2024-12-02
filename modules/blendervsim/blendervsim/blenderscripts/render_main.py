import os
import sys

# Add to the path to import local packages
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.manager import BlenderManager

def main():
    with BlenderManager() as manager:
        while manager.alive:
            manager.listen()


if __name__ == "__main__":
    main()
