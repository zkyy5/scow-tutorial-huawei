#!/bin/bash
set -o errexit

rm -rf ./release_web
mkdir ./release_web

rm -rf ./tar_tmp
mkdir -p ./tar_tmp/tutorial
cp -r Tutorial*/ ./tar_tmp/tutorial
cp -r tutorial_scow_for_ai.assets/ ./tar_tmp/tutorial
cd ./tar_tmp
tar -zcf tutorial.tar.gz tutorial/
cd ..

mv ./tar_tmp/tutorial.tar.gz ./release_web
rm -rf ./tar_tmp

cp -r ./tutorial_scow_for_ai.assets ./release_web
cp -r Tutorial*/ ./release_web

rm -rf ./release_pdf
mkdir -p ./release_pdf/tutorials/

process_directory() {
    local dir="$1"
    echo "Entering directory: $dir"
    find "$dir" -maxdepth 1 -type f -name "*.ipynb" | while read -r file; do
        mkdir -p "./release_web/${dir}"

        html_file="${file%.ipynb}.html"
        pdf_file="${file%.ipynb}.pdf"

        jupyter nbconvert --to html "$file"
        google-chrome --no-sandbox --headless --print-to-pdf="${pdf_file}" --no-pdf-header-footer --virtual-time-budget=10000 "$html_file"

        cp "${file}" "./release_web/${dir}"
        mv "${html_file}" "./release_web/${dir}"
        mv "${pdf_file}" "./release_pdf/tutorials/"
    done
}

export -f process_directory

# 把每个Tutorial下的文件转为html和pdf
find . -maxdepth 1 -type d -name "Tutorial*" | parallel process_directory

# 把根目录下的md文件转为html
python release_md.py

# 把根目录下的html文件转为pdf
find ./release_web/ -maxdepth 1 -type f -name "*.html" | while read -r html_file; do
    pdf_file="${html_file%.html}.pdf"
    google-chrome --no-sandbox --headless --print-to-pdf="${pdf_file}" --no-pdf-header-footer --virtual-time-budget=10000 "$html_file"

    mv "${pdf_file}" "./release_pdf/"
done

# pdf合并
find "./release_pdf/" -maxdepth 1 -type f -name "*.pdf" | while read -r pdf_file; do
    pdftk "${pdf_file}" ./release_pdf/tutorials/*.pdf cat output "${pdf_file}.merge.pdf"
    rm -rf "${pdf_file}"
    mv "${pdf_file}.merge.pdf" "${pdf_file}"
done
