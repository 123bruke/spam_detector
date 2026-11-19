# Mathematical notes: Bayes' theorem and the spam detector

## 1. Define the events

- `S`: the message is spam.
- `H`: the message is ham (legitimate).
- `M`: the observed message.

We want the posterior probability:

\[
P(S\mid M)
\]

That means: after observing the message, what is the probability that it belongs to the spam class?

## 2. Bayes' theorem

\[
\boxed{P(S\mid M)=\frac{P(M\mid S)P(S)}{P(M)}}
\]

The total-probability denominator for two classes is:

\[
P(M)=P(M\mid S)P(S)+P(M\mid H)P(H)
\]

Therefore:

\[
\boxed{P(S\mid M)=\frac{P(M\mid S)P(S)}{P(M\mid S)P(S)+P(M\mid H)P(H)}}
\]

### Four Bayes terms

| Term | Meaning in the spam detector |
|---|---|
| Prior `P(S)` | Spam fraction in the training data |
| Likelihood `P(M\|S)` | How likely the observed words are in spam messages |
| Evidence `P(M)` | Overall probability of observing those words |
| Posterior `P(S\|M)` | Updated probability that this message is spam |

Do not reverse the conditional probabilities: `P(M|S)` is not the same as `P(S|M)`.

## 3. Why Naive Bayes is called "naive"

A message can contain many tokens: `claim`, `prize`, `call`, `now`. A full joint probability over all possible token combinations would be difficult to estimate. Multinomial Naive Bayes makes a simplifying assumption that the tokens are conditionally independent once the class is known:

\[
P(M\mid S)\approx\prod_{i=1}^{V}P(w_i\mid S)^{n_i}
\]

- `V`: number of words/phrases in the learned vocabulary.
- `w_i`: one vocabulary token.
- `n_i`: how many times the token occurs in the message.

This assumption is not literally true for natural language, but it makes a useful, efficient baseline.

## 4. Laplace smoothing

A token that never appeared in spam training messages must not make the entire spam likelihood zero. Add a smoothing value `alpha`:

\[
P(w_i\mid S)=\frac{count(w_i,S)+\alpha}{N_S+\alpha V}
\]

`N_S` is the total number of token occurrences in spam training messages. The code uses `alpha = 0.5`.

## 5. Hand calculation (illustrative tiny example)

Suppose a toy training set gives:

- `P(spam) = 0.20`
- `P(ham) = 0.80`
- For a message with one observed token `prize`, assume `P(prize|spam) = 0.10`
- Assume `P(prize|ham) = 0.01`

The unnormalized scores are:

\[
Score_{spam}=0.10\times0.20=0.020
\]

\[
Score_{ham}=0.01\times0.80=0.008
\]

Normalize the scores:

\[
P(spam\mid prize)=\frac{0.020}{0.020+0.008}
=\frac{0.020}{0.028}\approx0.7143
\]

\[
\boxed{P(spam\mid prize)\approx71.43\%}
\]

This is a toy illustration, not a result from the UCI dataset. It shows why a rare spam-associated word may increase the posterior even when spam is less common overall.

## 6. How the code calculates a message score

The implementation stores class prior log-probabilities and token conditional log-probabilities. For class `c`:

\[
Score(c)=\log P(c)+\sum_i n_i\log P(w_i\mid c)
\]

Then it applies a two-class normalization:

\[
P(spam\mid M)=\frac{e^{Score(spam)}}{e^{Score(spam)}+e^{Score(ham)}}
\]

For token-level interpretation, the contribution of token `w` to spam-versus-ham log odds is:

\[
Contribution(w)=n_w\left[\log P(w\mid spam)-\log P(w\mid ham)\right]
\]

- Positive contribution: supports spam.
- Negative contribution: supports ham.

These are contributions under the Naive Bayes model, not proof that any one word is malicious.

## 7. Neural-network calculation

The second model follows a different learning method:

1. Convert the message into a TF-IDF numeric vector.
2. Pass the vector through fully connected layers with ReLU activations.
3. Produce one output logit `z`.
4. Convert it to a score using the sigmoid function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

5. Train weights using binary cross-entropy and backpropagation.

Unlike Naive Bayes, this MLP does not directly apply Bayes' theorem. It is included as a second model so that their held-out test performance can be compared.

## 8. Evaluation calculations

Let:
- TP = spam correctly identified as spam.
- TN = ham correctly identified as ham.
- FP = ham incorrectly flagged as spam.
- FN = spam incorrectly classified as ham.

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
Precision=\frac{TP}{TP+FP}
\qquad
Recall=\frac{TP}{TP+FN}
\]

\[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
\]

The test set is kept separate from model training. Use the test metrics, confusion matrices and ROC curves to assess generalization. Do not use training accuracy as the final performance claim.
