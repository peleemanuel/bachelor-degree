# Sistem de detectie si aplicatie livrabile

### Linkuri utile
- [Video Demo](https://youtu.be/dLwOLVX85vw)
- [Link Repo](https://github.com/peleemanuel/bachelor-degree)

### Structura fișierelor relevante
```.
├── app_interface
│   ├── app.py
│   ├── requirements.txt
│   ├── demo_overlap_algo.ipynb
│   ├── /raw_images
│   ├── /self_made_scripts
│   │   ├── add_info_to_images.py
│   │   └── (fișiere adiacente)
│   └── (fișiere adiacente)
├── deforestation_model
├── raspberry_code
└── (fișiere adiacente)
```

## Cerințe de sistem

- **Sistem de operare**: Ubuntu 20.04 sau versiuni superioare  
- **Micromamba** instalat pentru enviroment curat si dedicat rulării 
- **Python**: versiunea **3.12** sau mai nouă 

---

## Instalare și configurare mediu python

1. Creează mediul virtual cu micromamba:

   ```bash
   micromamba create -n drone-env python=3.12
   ```

2. Activează mediul creat

   ```bash
   micromamba activate drone-env
   ```

---

## Instalare dependențe

1. Schimbă directorul de lucru în folderul specific ```app_interface``` unde se află livrabilele aplicației
   ```bash
   cd app_interface
   ```

2. Pentru instalarea tuturor pachetelor necesare, folosiți fișierul requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

---

# Rulare livrabile

## Aplicația propriu zisă

Rulearea aplicației principale se face prin streamlit în folderul de lucru specificat mai sus.
   ```bash
   streamlit run app.py
   ```
Această comandă va porni local interfața web pentru interacțiunea cu sistemul de detecție. Pentru utilizarea acesteia, puteți consulta demo-ul live.

## Scripturi relevante

Pentru utilizarea scriptului ce adaugă informațiile GPS la imaginile capturate de dronă e necesar ca pre-requisite obținerea unui key API GOOGLE salvat într-un dotenv în interiorul folderului ```self_made_scripts```.

Script de adăugare date GPS în imagini este disponibil pentru inserarea informațiilor GPS în fișiere JPEG, folosind un fișier capture_gps_info.txt și fișiere de log generate în timpul zborului dronei.

### Utilizare
```bash
usage: add_info_to_images.py [-h] folder

Annotate JPEG captures with geolocation from GPS corrections.

positional arguments:
  folder      Path to the folder containing 'capture_gps_info.txt', 'run.log', and images.

options:
  -h, --help  show this help message and exit
```

### Exemplu de utilizare în interiorul repo-ului
Asigură-te ca te aflii în folderul ```app_interface/self_made_scripts``` după care poți rula pentru test următoarea comandă:
```bash
python add_info_to_images.py /home/epele/bachelor-degree/app_interface/raw_images/captures_2025_05_31_19_32
```

## Notebook util pentru vizualizarea algoritmului de detecție

E necesară instalrea urmatoarelor module:
```bash
micromamba install notebook ipykernel
```

Ulterior se poate deschide notebook-ul ```app_interface/demo_overlap_algo.ipynb``` și rulearea celulelor în ordinea respectivă.