# Attribution and third-party notices

The modelling specification and preprocessing design are based on the scikit-learn example **Tweedie regression on insurance claims**, inspired by Noll, Salzmann and Wüthrich's French MTPL case study. This project adds its own data audits, train-only preprocessing, evaluation/reporting implementation and sensitivity workflow. The upstream notice is retained below.

Source: https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING

```text
BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The CASdatasets authors are Christophe Dutang and Arthur Charpentier. Source data are not bundled in the version-controlled project. See `data/README.md` and `data/manifest.json` for provenance and the separate data terms.

The generated standalone HTML embeds the Plotly.js runtime distributed with Plotly.py; its upstream license header remains embedded in that runtime.

## Public synthetic insurance documents

Two original policy declaration PDFs and their extracted text in `data/text_demo/public/`, plus copies of that text in the recorded report/web payload, are from **Nikolai Sachok, Strata Insurance Corpus**. Source revision: `c9ad2b0b4cdc4041bda304390988ee8ce8386091`.

- Source: https://github.com/NikolaiSachok/strata-insurance-corpus
- Original paths: `sample/docs/policy/MOT-0000002-declarations.pdf` and `sample/docs/policy/MOT-0000004-declarations.pdf`.
- License for generated data and samples: **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/ . This data is not relicensed under this repository's BSD code license.
- Changes: original PDFs unchanged; text extracted using pypdf 6.1.3; questions, evaluation and UI added by this project. All persons, policies and organizations in those documents are synthetic. No endorsement is implied. Exact source URLs and checksums appear in `data/text_demo/manifest.json`.

## Pretrained neural text reader

**deepset/minilm-uncased-squad2**, by the deepset authors Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch and Tanay Soni. Based on Microsoft's MiniLM-L12-H384-uncased and fine-tuned on SQuAD 2.0. The model card marks the checkpoint **CC BY 4.0**: https://huggingface.co/deepset/minilm-uncased-squad2 . License: https://creativecommons.org/licenses/by/4.0/ . Used without fine-tuning; no endorsement is implied.

Pinned revision: `934656cdda79824eabf503ed56e15c01ddbdbe3f`. Weights are downloaded locally into an ignored cache as safetensors and are not bundled in this repository. Configuration, file hashes and measured extraction traces are published. The displayed text includes public CC BY 4.0 source content as described above.
