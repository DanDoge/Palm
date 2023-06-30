## Palm: Predicting Actions through Language Models @ Ego4D Long-Term Action Anticipation Challenge 2023. [arxiv](https://arxiv.org/abs/2306.16545)

###### Daoji Huang, Otmar Hilliges, Luc Van Gool, Xi Wang
<br>

This repo contains implementation of team Doggeee's solution to [Ego4D LTA challenge](https://eval.ai/web/challenges/challenge-page/1598/leaderboard/3881)@[3rd Ego4D Workshop CVPR 2023](https://ego4d-data.org/workshops/cvpr23/#).

Installing dependencies and preparing data should be done as in the official repo(see EGO4D_README.md) as followes.

Other dependencies
- EGOVLP video feature is obtained from [EGOVLP](https://github.com/showlab/EgoVLP) codebase, put the feature files into the respective folder should work.
  - then the usual training script should train a model predicting input actions, instead of future ones.
- transformers==4.27.2 and langchain==0.0.157 (or any version that has mmr examples selector)
  - I modified the langchain api to expose fetch_k and lambda_mult, but it turns out the out-of-box params work well, so delete these two if necessary

## Reproduce our results

We used blip2 with prefix "A person is" for caption generation by running 'python eval-blip2-caption.py'. 

And then our submission can be reproduced by 

```python
python eval-gpt-all.py --split test_unannotated --device 0 --nexample 8 --numbercaptions 1 --version v2 --caption_file  PATH_TO_CAPTION.json  --is_pred_action True --use_narration imagecaption --remove_duplicate True --max_nprev 12 --prompt_design maxmargin
```

## Citation

```
@article{huang2023palm,
  title={Palm: Predicting Actions through Language Models @ Ego4D Long-Term Action Anticipation Challenge},
  author={Huang, Daoji and Hilliges, Otmar and Van Gool, Luc and Wang, Xi},
  journal={arXiv preprint arXiv:2306.16545},
  year={2023}
}
```

