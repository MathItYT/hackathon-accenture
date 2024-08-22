from hackaton.backend import model
from hackaton.frontend import components
from hackaton.utils import environment


def main():
    environment.load_env()
    m = model.Model()
    c = components.get_component(m)
    c.launch()


if __name__ == "__main__":
    main()
