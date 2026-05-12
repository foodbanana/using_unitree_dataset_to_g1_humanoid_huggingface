#### 다운로드 위치 
```text
기본 저장 위치 : /mnt/hdd/huggingface/hub

저장 폴더 :
UnifoLM_G1_Brainco_Dataset
UnifoLM_G1_Dex1_Dataset
UnifoLM_G1_Dex1_DiverseManip_Dataset
UnifoLM_G1_Dex3_Dataset
UnifoLM-VLA-0
UnifoLM_WBT_Dataset
UnifoLM-WMA-0
UnifoLM_Z1_Arm_Dataset

```


#### .parquet 데이터 경로 예시
```text
"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


UnifoLM_G1_Brainco_Dataset : /mnt/hdd/huggingface/hub/UnifoLM_G1_Brainco_Dataset/datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset/snapshots/ad16b54daac9b12d815934723059048d4358ca1b/data/chunk-000/
UnifoLM_G1_Dex1_Dataset : /mnt/hdd/huggingface/hub/UnifoLM_G1_Dex1_Dataset/datasets--unitreerobotics--G1_Dex1_Bag_Insert/snapshots/ed1fd11a5f6b1e9df1e0cc0aaa319ceccedf6ab8/data/chunk-000
UnifoLM_G1_Dex1_DiverseManip_Dataset : Yet not exist(저는 이 데이터셋을 사용 안해서 다운 안받았습니다)
UnifoLM_G1_Dex3_Dataset : /mnt/hdd/huggingface/hub/UnifoLM_G1_Dex3_Dataset/datasets--unitreerobotics--G1_Dex3_BlockStacking_Dataset/snapshots/57faa2cf516e008f96d91fe3b67ad53a74f012e6/data/chunk-000
UnifoLM_WBT_Dataset: /mnt/hdd/huggingface/hub/UnifoLM_WBT_Dataset/datasets--unitreerobotics--G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/snapshots/16c01dbfcb2159783ea575acd42d1cec9b69e311/data/chunk-000/
```

#### 구체적인 폴더 내부 데이터셋 목록
```text
1.UnifoLM_G1_Brainco_Dataset
/mnt/hdd/huggingface/hub/UnifoLM_G1_Brainco_Dataset$ ls
datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset
datasets--unitreerobotics--G1_Brainco_GraspRubiksCube_Dataset
datasets--unitreerobotics--G1_Brainco_PickApple_Dataset
datasets--unitreerobotics--G1_Brainco_PickCharger_Dataset
datasets--unitreerobotics--G1_Brainco_PickDoll_Dataset
datasets--unitreerobotics--G1_Brainco_PickDrink_Dataset
datasets--unitreerobotics--G1_Brainco_PickTissues_Dataset
datasets--unitreerobotics--G1_Brainco_PickToothpaste_Dataset

2.UnifoLM_G1_Dex3_Dataset
/mnt/hdd/huggingface/hub/UnifoLM_G1_Dex3_Dataset$ ls
datasets--unitreerobotics--G1_Dex3_BlockStacking_Dataset
datasets--unitreerobotics--G1_Dex3_CameraPackaging_Dataset
datasets--unitreerobotics--G1_Dex3_GraspSquare_Dataset
datasets--unitreerobotics--G1_Dex3_ObjectPlacement_Dataset
datasets--unitreerobotics--G1_Dex3_PickApple_Dataset
datasets--unitreerobotics--G1_Dex3_PickBottle_Dataset
datasets--unitreerobotics--G1_Dex3_PickCharger_Dataset
datasets--unitreerobotics--G1_Dex3_PickDoll_Dataset
datasets--unitreerobotics--G1_Dex3_PickGum_Dataset
datasets--unitreerobotics--G1_Dex3_PickSnack_Dataset
datasets--unitreerobotics--G1_Dex3_PickTissue_Dataset
datasets--unitreerobotics--G1_Dex3_Pouring_Dataset
datasets--unitreerobotics--G1_Dex3_ToastedBread_Dataset

3.UnifoLM_WBT_Dataset
/mnt/hdd/huggingface/hub/UnifoLM_WBT_Dataset$ ls
datasets--unitreerobotics--G1_WBT_Brainco_Collect_Plates_Into_Dishwasher
datasets--unitreerobotics--G1_WBT_Brainco_Make_The_Bed
datasets--unitreerobotics--G1_WBT_Brainco_Pickup_Pillow
datasets--unitreerobotics--G1_WBT_Dex1_Put_Clothes_into_Washing_Machine
datasets--unitreerobotics--G1_WBT_Inspire_Collect_Clothes_MainCamOnly
datasets--unitreerobotics--G1_WBT_Inspire_Pickup_Pillow_MainCamOnly
datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_Into_Basket
datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_into_Washing_Machine
datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly
datasets--unitreerobotics--G1_WBT_Inspire_Put_Drinks_Into_Fridge
datasets--unitreerobotics--G1_WBT_Inspire_Put_Vegetables_Into_Basket

4. 나머지 데이터셋
나머지 데이터셋은 분석을 하지는 않기 때문에 생략했습니다.


```