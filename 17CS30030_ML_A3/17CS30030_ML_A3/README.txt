Keep all .py files and "AllBooks_baseline_DTM_Labelled.csv" in one folder

Then execute following command in given sequence
python3 A.py   # This will create two files modified dataset - "AllBooks_baseline_DTM_Labelled_modified.csv" and tfidf matrix = "mat.npy"  
python3 B.py   # This will create file "agglomerative.txt"
python3 C.py   # This will create file "kmeans.txt"
python3 D.py   # This will create files "agglomerative_reduced.txt" and "kmeans_reduced.txt"
python3 E.py   # This print NMI for each cluster on terminal


NMI for each cluster :  # This can very each time Due to randomization
	NMI for agglomerative is :  0.022844902439705637
	NMI for agglomerative_reduced is :  0.03724046182723285
	NMI for kmeans is :  0.3680962784682486
	NMI for kmeans_reduced is :  0.4527243430582473


