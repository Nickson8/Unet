# Setup Env Remote
- Uptade system and install other things
```bash
    apt update
    
    apt install nvim
    
    apt install zip
```

- Unzip dataset
```bash
    unzip dataset.zip
```

- Install poetry
```bash
    curl -sSL https://install.python-poetry.org | python3 -

    echo "export PATH='$HOME/.local/bin:$PATH'" >> ~/.bashrc

    poetry config virtualenvs.in-project true
```

- Install dependencies
```bash
    poetry init -n

    cat requirements.txt | xargs poetry add
```

# Commands in Local
- Uploading dataset
```bash
    make upload-file ./dataset.zip
```

- Download results
```bash
    make download-outputs ./results
```