import torch
import math

dtype = torch.float
device = torch.device("cpu")

letters = "abcdefghijklmnopqrstuvwxyz _"
NUM_LETTERS = 27
letter_to_idx = dict([(letters[i], i) for i in range(0, NUM_LETTERS)])

with open('big_words') as f:
    words = f.read().replace('-', '').split('\n')[:-1]

def letter_to_vec(letter):
    return torch.tensor(
        [letter_to_idx[letter]],
        dtype=torch.long,
        device=device,
    )


CONTEXT_SIZE = 30
def word_to_vec(in_word):
    train_word = in_word

    ww = " "*CONTEXT_SIZE
    ww = ww[len(train_word):] + train_word[:len(train_word)]

    vec = torch.tensor(
        [letter_to_idx[x] for x in ww],
        dtype=torch.long,
        device=device,
    )

    return vec


BATCH_SIZE = 5
train_batches = []
test_batches = []

for i in range(0, math.floor(len(words) / BATCH_SIZE), BATCH_SIZE):
    train_batch = []
    test_batch = []
    for w in words[i:i + BATCH_SIZE]:
        train_batch.append(word_to_vec(w[:-1]))
        test_batch.append(letter_to_vec(w[-1]))

    train_batches.append(torch.stack(train_batch))
    test_batches.append(torch.stack(test_batch))
    
#embeds = torch.nn.Embedding(26, 50)

#vec = torch.tensor([letter_to_idx[x] for x in words[0]], dtype=torch.long)


HIDDEN_SIZE = 128
class Test(torch.nn.Module):

    def __init__(self, embedding_dim, context_size):
        super(Test, self).__init__()
        self.embeddings = torch.nn.Embedding(NUM_LETTERS, embedding_dim, device=device)

        self.linear_key = torch.nn.Linear(embedding_dim*context_size, HIDDEN_SIZE, device=device)
        self.linear_query = torch.nn.Linear(embedding_dim*context_size, HIDDEN_SIZE, device=device)
        self.linear_value = torch.nn.Linear(embedding_dim*context_size, HIDDEN_SIZE, device=device)

        self.mha = torch.nn.MultiheadAttention(
            HIDDEN_SIZE,
            1,
            batch_first=True,
            device=device,
        )

        self.linear1 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device=device)

        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, NUM_LETTERS, device=device)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(BATCH_SIZE, 1, -1)

        key = self.linear_key(embeds)
        query = self.linear_query(embeds)
        value = self.linear_value(embeds)

        attn_output, attn_output_weights = self.mha(query, key, value)

        out = self.linear1(attn_output)
        out = torch.nn.functional.relu(out)
        out = self.linear2(out)

        log_probs = torch.nn.functional.log_softmax(out, dim=2)
        return log_probs


losses = []
loss_function = torch.nn.NLLLoss()
EMBEDDING_DIM = 50
model = Test(EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)


def blarg_old(a):
    for i in range(0, 5):
        vec = word_to_vec(a)
        out = model(torch.stack([vec for x in range(0, BATCH_SIZE)]))[0][0]
        nxt = torch.argmax(out)
        a = a + letters[nxt] 

    return a


def blarg(a, m):
    while len(a) == 0 or (a[-1] != ' ' and len(a) < CONTEXT_SIZE):
        vec = word_to_vec(a)
        out = m(torch.stack([vec for x in range(0, BATCH_SIZE)]))[0][0]
        nxt = torch.argmax(out)
        a = a + letters[nxt]

    return a

def sample(m, a):
    while len(a) == 0 or (a[-1] != ' ' and len(a) < CONTEXT_SIZE):
        vec = word_to_vec(a)
        out = m(torch.stack([vec for x in range(0, BATCH_SIZE)]))[0][0]
        nxt = torch.softmax(out, dim=-1)
        pick = torch.utils.data.sampler.WeightedRandomSampler(nxt, 1, replacement=False)
        for i in pick:
            a += letters[i]

    return a

def main():
    for epoch in range(20):
        print(f"Epoch {epoch}")
        total_loss = 0
        i = 0
        for j in range(0, len(train_batches)):
            i += 1
            if i % 999 == 0:
                print(i)

            # RSI

            model.zero_grad()
            log_probs = model(train_batches[j])
            loss = loss_function(log_probs.squeeze(), test_batches[j].squeeze())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)

    print(losses)
    torch.save(model.state_dict(), "mm.pt")
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
