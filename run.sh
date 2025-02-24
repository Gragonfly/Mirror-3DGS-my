#python train.py -s datasets/real/lounge -m output/real/lounge --eval -r 4 --convert_SHs_python --iteration 60000
#python train.py -s datasets/lounge -m output/lounge --eval -r 4 --convert_SHs_python --iteration 60000
#python train.py -s datasets/real/discussion_room -m output/real/discussion_room --eval -r 4 --convert_SHs_python --iteration 60000
#python train.py -s datasets/real/market -m output/real/market --eval -r 4 --convert_SHs_python --iteration 60000
#python train.py -s datasets/synthetic/office -m output/synthetic/office --eval -r 2 --convert_SHs_python --iteration 60000
python train.py -s datasets/synthetic/washroom -m output/synthetic/washroom --eval -r 2 --convert_SHs_python --iteration 60000
python train.py -s datasets/synthetic/livingroom -m output/synthetic/livingroom --eval -r 2 --convert_SHs_python --iteration 60000

#python train.py -s datasets_blender/real/discussion_room -m output_blender1/real/discussion_room --eval -r 4 --convert_SHs_python --iteration 60000
#python train.py -s datasets_blender/synthetic/office -m output_blender/synthetic/office --eval -r 2 --convert_SHs_python --iteration 60000


# discussion_room and office  images too few