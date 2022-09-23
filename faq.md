# FAQ
---
### How do I use objects instead of vectors?
1) Set archive to use objects:


```yaml
# -- Archive -- #
archive:
  type: "Grid"
  grid_res: [20,20]
  desc_bounds:
  - [-2, 2]
  - [-2, 2]
  desc_labels:
  - Param 1
  - Param 2  
  use_objects: True # <----- LIKE THIS ----|
```

2) Set to the `Object` emitter, along with relevant mutation parameters (mutation must be defined in the object). Here is a CPPN:

```yaml
# -- Emitters -- #
emitters:
  -
    name: "CPPN"
    type: "Object"  # <----- Object Emitter ----|
    n_batch   : 75
    # -- Mutation -- # 
    mut_p:          # <----- Mutation hyperparameters ----|
      add_conn   : 0.15
      add_node   : 0.15
      rem_conn   : 0.15
      rem_node   : 0.15
      mod_conn   : 0.15
      mod_node   : 0.15
      mod_sigma  : 0.05
      act_fcns   : [1,2,3,4,5,6,7]
      weight_cap : 1
      param_bounds: [-1, 1]
```

---
