# EPIC-KITCHEN VISOR Pre-Processing Scripts

## Config file
Use `config.yml` to control how to filter objects. 
- `coverage_filter` field is to filter objects that are too small or too large, based on either bounding boxes or mask coverage (value of mode, `bbox` or `mask`).
- `static_filter` field is to filter objects based on its category or name. 

## Run
### To get only filtered annotations in the same format with EPIC-VISOR:
```python
python filter_hand_object.py -s [set] -o [output]
```
`set` can be either `train`, `val`, or `both`
E.g:
```python
python filter_hand_object.py -s both -o .
```
### To get dataset in DAVIS format (default resolution is `854x480`)
- Hand-held objects only
```python
python DAVIS_generate_trajectory.py -s [set] -o [output] -r [resolution]
```
E.g:
```python
python DAVIS_generate_trajectory.py -s val -o . 
```
- Hand-held objects with hands
```python
python DAVIS_generate_trajectory.py -s [set] -o [output] -r [resolution] -hand
```
E.g:
```python
python DAVIS_generate_trajectory.py -s val -o . -hand
```
