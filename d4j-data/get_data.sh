pid=$1
vid=$2
scp ubuntu:/home/bohrok/Documents/defects4j-coverage-matrix/single/$pid-$vid.pkl covmat
scp ubuntu:/home/bohrok/Documents/defects4j-coverage-matrix/result/bugdata/$pid-$vid.json bugdata