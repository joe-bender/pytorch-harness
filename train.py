def train(n_hidden, hidden_size, epochs, trainX, trainy, testX, testy):
    model = Net(n_hidden, hidden_size)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=.09)
    
    log = {'train_loss': [],
           'train_metric': [],
           'test_loss': [],
           'test_metric': []}
    
    for _ in range(epochs):
        # train
        model.train()
        optimizer.zero_grad()
        
        outputs = model(trainX)
        loss = criterion(outputs, trainy)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_metric = get_accuracy(outputs, trainy)

        # validate
        model.eval()
        with torch.no_grad():
            outputs = model(testX)
            loss = criterion(outputs, testy)
        
        test_loss = loss.item()
        test_metric = get_accuracy(outputs, testy)
        
        # log
        log['train_loss'].append(train_loss)
        log['train_metric'].append(train_metric)
        log['test_loss'].append(test_loss)
        log['test_metric'].append(test_metric)
    
    return log