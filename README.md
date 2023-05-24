# OC-project_7

Le projet s'inscrit dans la formation "Data Science" dispensé par l'organiseme de formation OpenClassroom en partenariat avec l'école d'ingénieur CentraleSupélec.
L'objectif de ce projet est de déployer une application de credit scoring sur le Web en utilisant, le framework Flask pour la partie API et le framwork 
streamlit pour la partie Dashboard. 
Le modèle de machine learning utilisé renvoi une probabilité de défaut, la modélisation est décrite dans notebook "scoring_model.ipynb". Les input de ce modèle sont les informations du client (salaire, nombre d'enfant à charge, sexe ... )
Si la probabilité en sortie du modèle dépasse un certain seuil, le crédit n'est pas accordé au client. Le seuil est fixé selon une contrainte métier basée sur le raisonnement suivant :
Il est plus risqué d'attribuer un crédit à un client non solvable plutôt que de refuser un crédit à un client solvable. 
L'application a été déployé grâce à la plateforme Heroku, il s'agissait d'une version gratuite à l'époque ce qui n'est plus le cas aujourd'hui.
