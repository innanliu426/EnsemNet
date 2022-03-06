from keras.callbacks import EarlyStopping
usualCallback = EarlyStopping()
overfitCallback = EarlyStopping(monitor='loss', mode='min', min_delta=0.05, patience = 2)


def efficienct_train(models, x_train, y_train, epoch_count, lower_bound, \
                     marker, psnr_value, filename, max_epoch, lower_bound_min, validate_file_loc):
    while ((lower_bound >=lower_bound_min) and (epoch_count<=max_epoch)):
        train = models.fit(x_train, y_train, batch_size=32,epochs=10,verbose=1, shuffle=True, callbacks=[overfitCallback])
        if epoch_count%25 == 0:
            ####### avoid overtrain, stop when testing psnr drops
            if train.history['loss'][-1] < marker:
                list_files_HR = glob(os.path.join(validate_file_loc, "*.png"))
                p = predict_this_set(models,list_files_HR[1:4], version = version, remainder=2) 
                if p > psnr_value:
                    psnr_value = p
                    marker = train.history['loss'][-1] 
                    models.save(filename) 
        ###### if exploded, read from saved version
        if ((train.history['loss'][-1] > 10e3) and (epoch_count>=50)):
            models = load_model(filename,custom_objects={'tf':tf})
            epoch_count -= 10
        ###### update threshold
        if train.history['loss'][-1]<=lower_bound:
            lower_bound = train.history['loss'][-1]
        epoch_count += train.params['epochs']
        print(epoch_count)
    
    if train.history['loss'][-1] < marker:
        list_files_HR = glob(os.path.join('../Images/Set5/', "*.png"))
        p = predict_this_set(models,list_files_HR[1:4], version = version, remainder=2)
        if p > psnr_value:
            psnr_value = p
            marker = train.history['loss'][-1] 
            models.save(filename) 
    
    ####  output testing result directly
    models = load_model(filename,custom_objects={'tf':tf})
    list_files_HR = glob(os.path.join('../Images/Set5/', "*.png"))
    _ = predict_this_set(models,list_files_HR, version = version, remainder=2)
    list_files_HR = glob(os.path.join('../Images/Set14/', "*.png"))
    _ = predict_this_set(models,list_files_HR, version = version, remainder=0)
