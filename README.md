# Network Analysis on Energy Data (CPFL)  
This repository implements a workflow to **build and analyze a network** based on energy data from **CPFL** (a Brazilian energy company that serves the interior of São Paulo state).  
The analysis is inspired by concepts from **Albert-László Barabási's book _Network Science_**, focusing on understanding the structure, connectivity, and properties of networks derived from real-world energy consumption data.

---

## 📌 Objectives
- **Process raw energy data** into structured formats for analysis.
- **Build and visualize networks** to explore relationships and dependencies.
- **Generate insights** from the perspective of complex networks theory.
- Provide **reproducible scripts** for data processing and network visualization.

---

## 📂 Project Structure
```
project-root/
│
├── data/
│   ├── bronze/              # Original datasets (read-only; not tracked if large/sensitive)
│   ├── silver/              # Cleaned/normalized intermediate datasets
│   └── gold/                 # Aggregates, network metrics, exports for reports
│    
├── src/
│   ├── utils/
│   │    ├── data_source.py      # DataSource class
│   │    └── segment_merger.py   # SegmentMerger class
│   │
│   └── a_build_network.py   # Merges/derives segments, builds graphs, saves to
│                             # data/gold/
├── visualization/       # Plots, network images, dashboard-ready assets

├── requirements.txt      # Python dependencies (see section below)
└── README.md             # This file
```


---

## ⚙️ Requirements
- **Python:** `3.10.12`
- **Virtual Environment:** Recommended using `virtualenvwrapper` (`mkvirtualenv` command)

### Main Libraries
- `pandas`
- `geopandas`
- `matplotlib`

See full list in [requirements.txt](./requirements.txt).

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone git@github.com:yanprada/mo412.git
cd mo412
```

### 2. Create a virtual environment
```bash
mkvirtualenv network-cpfl -p python3.10.12
```

### 3. Activate the environment
```bash
source network-cpfl/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Install setup
```bash
python setup.py install
```
---

## ▶️ Usage

### Step 1: Download the raw data

- Download the raw data from https://dadosabertos-aneel.opendata.arcgis.com/datasets/79071ab68be94f6f91b5c2eead4e2384/about

- Save it in `data/bronze` folder.

### Step 2: Build and analyze the network

Run the second script to:

- Merge processed data

- Create network graphs

- Save outputs in `data/silver/`, `data/gold/` and `visualization/` folders

```bash
python src/a_build_network.py
```
---

## 📊 Outputs
- Intermediate Processed datasets: `data/silver/`

- Network data: `data/gold/`

- Graph visualizations: `visualization/`

---

## 🔍 References

- Albert-László Barabási, Network Science: http://networksciencebook.com

- CPFL Energia (raw data): https://dadosabertos-aneel.opendata.arcgis.com/datasets/79071ab68be94f6f91b5c2eead4e2384/about

- CPFL Energia (metadata): https://dadosabertos-aneel.opendata.arcgis.com/documents/f0d5c43ac67d4f5eb2ddffa4589501b2/explore
---

## ✅ Reproducibility Tips

- Keep Python at 3.10.12 for consistent dependency behavior.

- Pin dependencies (use the provided requirements.txt).