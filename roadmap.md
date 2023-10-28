# Plans

* Refactor Model to include tokenization, embedding and position encoding.

* Refactor Configuration and database to use the new Model.

* Once we have position embedding make a variant of `suffix` that adds together values from positions, rather than stacking them. There will be two variants: `suffix.X.add` and `suffix.X.stack`

* Same for Attention.

* Refactor the database to use the same instance for several machines. (Instead of adding extra DBs in the prediction model).

* 