Description des fichiers dans le dossier **metrics**

# append_summary.py 

- Programme python : Rajoute en fin de fichier (ou crée le fichier) un résumé de tous les jobs lancés sur le cluster, il y a un csv par utilisateur, et les jobs sont triés par date de début croissante (les derniers jobs sont en bas du fichier)

# slurm_all.slurm & slurm_daily.slurm

- Scripts slurm : Permet de lancer le programme python sur le cluster, renvoie les logs dans le dossier logs
- Daily pour les jobs d'il y a 2 jours à il y a 1 jour (de -48h à -24h) on ne prend pas les jobs qui viennent d'être lancés car on aura pas forcément le taux de co2 pour un moment trop récent.
- All pour tous les jobs lancés sur le cluster, jusqu'à -24h.
- Possibilité de changer les dates mesurées en modifiant les valeurs "days" ici :
```python
    start_time = (datetime.datetime.now() - datetime.timedelta(days=70)).timestamp()
    end_time = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
```
(par exemple ici on prend de -70j à -1j)

# get_co2.py & co2.json

- Programme python qui fait appel à l'api de "https://odre.opendatasoft.com" pour récupérer les données de co2, et les stocke dans un fichier json.

# clean_jobs.sh & check_wrong.sh

- Check : Parcours le dossiers d'où proviennent les logs que l'on génère par le programme python, et vérifie si les jobs récupérés sont corrects.
- Clean : Même chose que check, mais supprime les jobs incorrects au lieu de les lister.

# cron_script.sh 

- Modification du crontab actuel pour lancer correctement les programme python.

# profile_d_print.sh | Outdated à priori (se référer à liastatus.sh)

- Script qui permet d'afficher sur terminal les derniers jobs lancés par l'utilisateur, a pour but d'être lancé à chaque login.
- Attention en copiant il faut peut être modifier USER tout en haut par "$USER" (normalement à dé commenter), pour que le script prenne le nom de l'utilisateur courant, il se peut que du debug soit fait avec d'autres utilisateurs écrit en dur pour tester.

# detail_csv.sh

- Script qui permet d'afficher des informations supplémentaires sur les données des CSV, se veut "plus complet" que profile_d_print.sh
- Le lancer avec -h ou --help pour avoir les options disponibles
- Idem pour le $USER

# liastatus.sh

- Copie de /bin/liastatus.sh en rajoutant la fonction de profile_d_print.sh
- à la fin, j'affiche ça "If you wish to have more informations about your jobs, you can use the following command : <TODO> -h"
- Le TODO l'idéal serait d'ajouter une commande comme ça a déjà été fait avec "lab-ia-doc", qui serait "detail_csv.sh"

# Informations dont il faut tenir compte :

- Pour que le programme de récupération des données dans les csv (append_summary.py) tourne correctement, il faut compiler **ma** version de AIPowerMeter, qui est dans mon home. (saesvin), j'ai travaillé avec un environnement conda (que j'ai appelé labia).
- 