# diplomka

- 	Odhad disparity se spouští kódem predict.py

- 	Je nutné v něm nastavit cestu k levému a pravému obrazu a disparitě.
	Poté cestu k natrénovanému modelu.pth.
	Nakonec se musí nastavit, jaký typ obrázků budeme odhadovat v proměnné "dataset".
	Každý dataset se musí načítat trochu jinak.
	Typy jsou rozdělené podle mých použitých datasetů. takže "SceneFlow", "DrivingStereo", "KITTI" a "Tramvaj".
	Pokud chcete použít snímky z jiného datasetu, který má ale stejný formát jako některý ze zmíněných datasetů, zapište tam ten

- 	Poté spusťte predict.py. Vykreslí vám odhadnutou disparity, orig. obrázek, orig. disparitu a absulutní chybu mezi odhadnutou a originální disparitou.
	do konzole také vypíše výsledky metrik EPE, RMSE a 3PE.

-	Ve složce "checkpoints" u každé neuronové sítě přikládám natrénoaný model pro každý dataset


BONUS INFO:	predict.py funguje vždy jen pro jeden snímek, ne pro celý dataset.
		Důležité knihovny viz requirements.txt
