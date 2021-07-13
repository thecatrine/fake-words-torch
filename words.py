import torch
import math

dtype = torch.float
device = torch.device("cpu")

letters = "abcdefghijklmnopqrstuvwxyz "
letter_to_idx = dict([(letters[i], i) for i in range(0, 27)])

with open('words.txt') as f:
    temp_words = f.read().split('\n')[:-1]

words = []
for temp in temp_words:
    words = words + [temp[:i] for i in range(1, len(temp)+1)]

#embeds = torch.nn.Embedding(26, 50)

#vec = torch.tensor([letter_to_idx[x] for x in words[0]], dtype=torch.long)


class Test(torch.nn.Module):

    def __init__(self, embedding_dim, context_size):
        super(Test, self).__init__()
        self.embeddings = torch.nn.Embedding(27, embedding_dim)
        self.linear1 = torch.nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = torch.nn.Linear(128, 27)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))

        out = self.linear1(embeds)
        out = torch.nn.functional.relu(out)
        out = self.linear2(out)

        log_probs = torch.nn.functional.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = torch.nn.NLLLoss()
EMBEDDING_DIM = 50
CONTEXT_SIZE = 20
model = Test(EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def letter_to_vec(letter):
    return torch.tensor(
        [letter_to_idx[letter]],
        dtype=torch.long,
        device=device,
    )


def word_to_vec(in_word):
    train_word = in_word

    ww = " "*20
    ww = ww[len(train_word):] + train_word[:len(train_word)]

    vec = torch.tensor(
        [letter_to_idx[x] for x in ww],
        dtype=torch.long,
        device=device,
    )

    return vec


def blarg(a):
    for i in range(0, 5):
        vec = word_to_vec(a)
        nxt = torch.argmax(model(vec))
        a = a + letters[nxt]

    return a


for epoch in range(10):
    print(f"Epoch {epoch}")
    total_loss = 0
    i = 0
    for word in words:
        i += 1
        if i % 999 == 0:
            print(i)

        ww = word_to_vec(word[:-1])
        test_ww = letter_to_vec(word[-1])

        model.zero_grad()
        log_probs = model(ww)
        loss = loss_function(
            log_probs,
            test_ww,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)

print(losses)
import pdb;pdb.set_trace()