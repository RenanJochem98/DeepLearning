from ClassificadorBinario import ClassificadorBinario

if __name__ == '__main__':
    entradas_file_name = "datasets/breast_cancer/entradas.csv"
    saidas_file_name = "datasets/breast_cancer/saidas.csv"

    c = ClassificadorBinario(entradas_file_name, saidas_file_name);
    resultados = c.predict()
    c.show_results(resultados)
