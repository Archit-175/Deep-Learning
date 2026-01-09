# Contributing Guidelines

## 📝 How to Add New Lab Assignments

### For Each New Lab:

1. Navigate to the respective lab folder (Lab01, Lab02, etc.)

2. Update the README.md with:
   - Assignment title
   - Objectives
   - Tasks to complete
   - Implementation details
   - Results and observations

3. Add your code files:
   - Python scripts (`.py` files)
   - Jupyter notebooks (`.ipynb` files)
   - Any supporting files

4. Create a results folder if needed:
   ```bash
   mkdir LabXX/results
   ```

5. Add visualizations, plots, or output files to the results folder

### Code Style Guidelines

#### Python Code
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Include comments for complex logic

Example:
```python
def train_model(X_train, y_train, epochs=10):
    """
    Train the neural network model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    # Your implementation here
    pass
```

#### Jupyter Notebooks
- Add markdown cells to explain each section
- Include visualizations
- Show results and analysis
- Clear output before committing (optional)

### File Organization

```
LabXX/
├── README.md           # Assignment description
├── main.py            # Main implementation
├── utils.py           # Utility functions (if needed)
├── model.py           # Model definitions (if needed)
├── results/           # Output files
│   ├── plots/        # Graphs and visualizations
│   └── logs/         # Training logs
└── data/             # Sample data (if small)
```

### Documentation

Each lab should include:
- Clear problem statement
- Solution approach
- Code explanation
- Results interpretation
- Key learnings

### Version Control

- Make meaningful commit messages
- Commit regularly
- Don't commit large data files (use .gitignore)
- Don't commit model checkpoints unless necessary

### Testing

- Test your code before committing
- Ensure all dependencies are listed in requirements.txt
- Verify that code runs on a fresh environment

## 🐛 Reporting Issues

If you find any issues or have suggestions:
1. Open an issue in GitHub
2. Describe the problem clearly
3. Provide steps to reproduce (if applicable)

## 💡 Suggestions

Feel free to suggest:
- Better code organization
- Additional utilities
- Documentation improvements
- New lab topics

---
Happy Coding! 🚀
