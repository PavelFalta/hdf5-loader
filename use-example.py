from loader import SingleFileExtractor, FolderExtractor

# Cesta k souboru nebo složce s daty
path = "/media/DATA/data/DATASETS/2024-03-04/dataset_0"

# Vytvoření instance extraktoru pro složku
extractor = FolderExtractor(path)

# Načtení surových dat pro 'icp'
data = extractor.get_raw_data('icp')

# Automatické anotování dat ve složce
extractor.auto_annotate("/media/DATA/data/DATASETS")

# Výpis popisu do souboru 'describe.txt'
print(extractor.describe("describe.txt"))

# Získání seznamu souborů
print(extractor.get_files())

# Získání seznamu módů
print(extractor.get_modes())

# Získání seznamu anotátorů pro 'icp'
print(extractor.get_annotators("icp"))

# Extrakce dobrých dat a anotací pro 'icp'
good, ann = extractor.extract("icp")

# Načtení dat z anotací
extractor.load_data(good, ann)

# Výpis tvaru dat první anotace
print(ann[0].data.shape)

# Výpis prvních pěti anotací
print(ann[:5])
