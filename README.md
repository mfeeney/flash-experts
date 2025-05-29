# Flash-Experts: GPT-2 with Flash Attention and Mixture of Experts

This project implements and experiments with modern machine learning architecture innovations by adding Flash Attention and Mixture of Experts (MoE) to GPT-2. The goal is to understand and implement these techniques while learning about modern ML architecture.

## Project Structure

```
flash-experts/
├── src/
│   ├── flash_attention/    # Flash Attention implementation
│   └── moe/               # Mixture of Experts implementation
├── tests/                 # Unit tests
├── data/                  # Dataset storage
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks for experiments
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Features (Planned)

- [ ] Flash Attention implementation
- [ ] Mixture of Experts implementation
- [ ] GPT-2 fine-tuning pipeline
- [ ] Performance benchmarking
- [ ] Training and inference scripts

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Status

This project is currently in early development. The implementation will proceed in the following order:

1. Set up basic GPT-2 fine-tuning pipeline
2. Implement Flash Attention
3. Implement Mixture of Experts
4. Benchmark and optimize

## Hardware Requirements

- CPU: 16 cores (Intel Xeon @ 2.30GHz)
- RAM: 58GB
- GPU: To be determined (currently CPU-only)

## License

MIT License

## Contributing

This is a learning project. Feel free to fork and experiment!
