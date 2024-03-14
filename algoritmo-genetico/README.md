# UFSJ-Bioinspirados

├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── tests
│   ├── __init__.py
│   ├── test_base.py
│   └── test_examples.py
├── examples
│   ├── __init__.py
│   ├── binary_function_otimizer.py
│   └── traveling_salesman.py
└── src
    ├── __init__.py
    ├── base.py
    ├── selection
    │   ├── __init__.py
    │   ├── abstract_selection.py  (New)
    │   ├── binary_tournament.py  (Move to selection)
    │   └── roulette_wheel.py     (Move to selection)
    ├── mutation
    │   ├── __init__.py
    │   ├── abstract_mutation.py  (New)
    │   ├── mutate_pos.py           (Move to mutation)
    │   └── mutate_ind.py          (Move to mutation)
    ├── crossover
    │   ├── __init__.py
    │   ├── abstract_crossover.py (New)
    │   ├── ox_cross.py             (Move to crossover)
    │   └── point_cross.py          (Move to crossover)
    └── utils.py