
# SEAT

This repository is an Pytorch implementation of paper: "SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries".


Note: this is not the official implementation of SEAT, you can follow the paper here: [https://dl.acm.org/doi/10.1145/3474369.3486863](https://dl.acm.org/doi/10.1145/3474369.3486863).


![Illustration of detection schemes of SEAT.](https://raw.githubusercontent.com/grasses/SEAT/master/exp/figure1.png)

<br>

# Dependencies

The code requires dependencies that can be installed using the `pip` environment file provided:
```
pip install -r requirements.txt
```

# Usage

Run main.py to fine-tune encoder and then evaluate SEAT
```
python3 main.py
```

Download fine-tuned encoder here: [https://drive.google.com/drive/folders/1RgeDjPNs9Tswn7hmkzBLLSl8mRJxBFm4?usp=sharing](https://drive.google.com/drive/folders/1RgeDjPNs9Tswn7hmkzBLLSl8mRJxBFm4?usp=sharing)


# License

This library is under the GPL V3 license. 
For the full copyright and license information, please view the [LICENSE](https://raw.githubusercontent.com/grasses/SEAT/master/LICENSE) file that was distributed with this source code.