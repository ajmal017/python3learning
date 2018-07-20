

def tar():
    import os, zipfile, shutil
    def zip_file(source_path, target_path, remove=False):
        """"""
        with zipfile.ZipFile(target_path, 'w') as zipf:
            pre_len = len(os.path.dirname(source_path))
            for parent, dirnames, filenames in os.walk(source_path):
                for filename in filenames:
                    pathfile = os.path.join(parent, filename)
                    arcname = pathfile[pre_len:].strip(os.path.sep)
                    zipf.write(pathfile, arcname)
        if remove:
            shutil.rmtree(source_path)

    def unzip_file(source_file, target_path, remove=False):
        """"""
        with zipfile.ZipFile(source_file) as zipf:
            for names in zipf.namelist():
                zipf.extract(names, target_path)

        if remove:
            os.remove(source_file)

    source = '/data/mapleleaf/work/algorithm/model/DecisionTreeClassifier_node_DecisionTreeClassifier_76017b.None'
    zipf = '/data/mapleleaf/work/algorithm/model/DecisionTreeClassifier_node_DecisionTreeClassifier_76017b.None.zip'
    zip_file(source, zipf, False)
    print(os.path.getsize(zipf))

if __name__ == '__main__':
    tar()