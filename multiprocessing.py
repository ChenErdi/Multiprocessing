import logging as log
import numpy as np
import gvc.data_structures
import gvc.binarization
import gvc.sort
import gvc.common
import multiprocessing as mp
import time
import collections
import copy as cp

def worker_reader(input_f,block_size,queue_reader,lock_reader,num_encoder,buffer_size=1): 
    
    log.info('Reader started')

    block_ID = 0
    lines = []
    is_eof = False
    
    while not is_eof:  
        line = input_f.readline()

        if len(line):
            lines.append(line)   
        else:
            is_eof = True
        # Put lines with block ID into a queue 
        if len(lines) == block_size or (is_eof and len(lines) > 0):
            wait = True
            while wait:
                lock_reader.acquire()
                try:
                    if queue_reader.qsize() < buffer_size:
                        block = [block_ID,cp.copy(lines)]
                        queue_reader.put(block)
                        log.info("Reader put block_ID:{} with size {} into queue_reader".format(block_ID,len(lines)))

                        lines.clear()
                        block_ID += 1 
                        wait = False
                finally:
                    lock_reader.release()
    # Send the stop signal to all worker encoder
    for i in range(num_encoder):
        queue_reader.put('Stop')       
    log.info("worker_reader is finished.")

def worker_encoder(
    binarizer,
    encoder,
    axis,
    sort_rows,
    sort_cols,
    transpose,
    block_size,
    max_cols,
    dist,
    preset_mode,
    queue_reader,
    queue_encoder,
    lock_reader,
    lock_encoder
):
    log.info('Worker Encoder started')
    running = True
    while running:
        get_data = True
        while get_data:
            lock_reader.acquire()
            try:
                if queue_reader.qsize():
                    get_reader_data = queue_reader.get()
                    get_data = False
            finally:
                lock_reader.release()

        if get_reader_data == 'Stop':
            # Put Stop signal to worker writer
            lock_encoder.acquire()
            try:
                queue_encoder.put('Stop')
            finally:
                lock_encoder.release()
            log.info('worker encoder get stoping sinal')
            running = False
        else:
            curr_block_ID = get_reader_data[0]
            lines = get_reader_data[1]
            log.info("Worker Encoder got current block ID: {}".format(curr_block_ID))

            # Excecute part 4.1 - splitting genotype matrix
            if len(lines) == 0:
                raise ValueError("lines from worker encorder is empty")

            log.info('Execute part 4.1 - Splitting genotype matrix')

            allele_matrix, phasing_matrix, p = gvc.binarization.split_genotype_matrix(lines)
            allele_matrix, any_missing, not_available = gvc.binarization.adaptive_max_value(allele_matrix)

            # Execute part 4.2 - binarization of allele matrix
            log.info('Execute part 4.2 - Binarization')
            bin_allele_matrices, additional_info = gvc.binarization.binarize_allele_matrix(
                allele_matrix, 
                binarizer=binarizer, 
                axis=axis
            )

            # Create parameter based on binarization and encoder parameter
            log.info('Create parameter set')
            new_param_set = gvc.common.create_parameter_set(
                any_missing,
                not_available,
                p,
                phasing_matrix,
                additional_info,
                binarizer,
                encoder,
                axis,
                sort_rows,
                sort_cols,
                transpose,
            )

            # Execute part 4.3 - sorting
            log.info('Execute part 4.3 - Sorting')
            sorted_data = gvc.sort.sort(
                new_param_set, 
                bin_allele_matrices, 
                phasing_matrix, 
                dist=dist, 
                time_limit=gvc.lkh.preset_time[preset_mode],
                max_cols = max_cols
            )
            # Execute part 4.4 - entropy coding
            log.info('Execute part 4.4')
            data_bytes = gvc.entropy.encode(new_param_set, additional_info, *sorted_data)

            # Initialize EncodedVariant, ParameterSet is not stored internally in EncodedVariants
            # Order of arguments here is important (See EncodedVariants)
            enc_variant = gvc.data_structures.EncodedVariants(new_param_set, *data_bytes)

            # Create new Block and store
            block = gvc.data_structures.Block.from_encoded_variant(enc_variant)    
            combined_data = (curr_block_ID, new_param_set, block)
            lock_encoder.acquire()
            try:
                queue_encoder.put(combined_data)
                log.info("Already put block ID: {} in queue_encoder".format(curr_block_ID))
            finally:
                lock_encoder.release()
            del combined_data
    
    
def worker_writer(
    output_fpath,
    binarizer,
    encoder,
    axis,
    sort_rows,
    sort_cols,
    transpose,
    block_size,
    max_cols,
    dist,
    preset_mode,
    queue_encoder,
    lock_encoder,
    num_encoder
):
    log.info("Worker writer start")

    acc_unit_param_set = None  # Act as pointer, pointing to parameter set of current AccessUnit
    acc_unit_id = 0
    blocks = []
    param_sets = []

    max_num_blocks_per_acc_unit = 2**(gvc.data_structures.AccessUnitHeader.NUM_BLOCKS_LEN * 8) - 1

    writer_dict = dict()
    get_block_ID = []
    new_param_set = []
    block = []

    catch_new_block = True
    writer_block_ID = 0
    count_stop_signal= 0
    running = True
    
    with open(output_fpath, 'wb') as output_f:

        while running:

            if catch_new_block:
                get_data = True

                while get_data:
                    lock_encoder.acquire()
                    try:
                        if queue_encoder.qsize():
                            get_encoder_data = queue_encoder.get()
                            get_data = False
                    finally:
                        lock_encoder.release()

                if get_encoder_data == 'Stop':
                    count_stop_signal += 1
                    log.info("Writer got {} stop signal".format(count_stop_signal))
                    # when all the stop signal which from encoder received, worker writer will stop
                    if count_stop_signal == num_encoder:
                        catch_new_block = False
            
                else:
                    # get the result from encoder
                    get_block_ID = get_encoder_data[0]
                    new_param_set = get_encoder_data[1]
                    block = get_encoder_data[2]
                    writer_dict[get_block_ID] = [new_param_set, block]
                
            # write file in correct order
            if writer_block_ID in writer_dict:
                log.info("Worker Writer is writting block_ID:{}".format(writer_block_ID)) 
                combined_data = writer_dict[writer_block_ID]
                new_param_set = combined_data[0]
                block = combined_data[1]
                del writer_dict[writer_block_ID]
                writer_block_ID += 1
                # If parameter set of current block different from parameter set of current access unit,
                # store blocks as access unit
                if acc_unit_param_set is None:
                    log.info('Parameter set of access unit is None -> set to new parameter set')
                    acc_unit_param_set = new_param_set
                    param_sets.append(acc_unit_param_set)
                    output_f.write(acc_unit_param_set.to_bytes())

                elif new_param_set != acc_unit_param_set or len(blocks) == max_num_blocks_per_acc_unit:
                    log.info('Parameter set and parameter set of current access unit is different')

                    # Store blocks as an Access Unit
                    log.info('Store access unit ID {:03d}'.format(acc_unit_id))
                    gvc.common.store_access_unit(output_f, acc_unit_id, acc_unit_param_set, blocks)

                    # Initialize values for the new AccessUnit
                    acc_unit_id += 1
                    blocks.clear()

                    # Check if similar parameter set is already created before                       
                    is_param_set_unique = True
                    for stored_param_set in param_sets:
                        if stored_param_set == new_param_set:
                            is_param_set_unique = False
                            break
                    
                    # If parameter set is unique, store in list of parameter sets and store in GVC file
                    if is_param_set_unique:
                        log.info('New parameter set is unique')
                        new_param_set.parameter_set_id = len(param_sets)

                        acc_unit_param_set = new_param_set

                        param_sets.append(acc_unit_param_set)
                        output_f.write(acc_unit_param_set.to_bytes())

                    else:
                        log.info('New parameter set is not unique')
                        del new_param_set
                        acc_unit_param_set = stored_param_set

                blocks.append(block)
            
            if len(writer_dict) == 0 and not catch_new_block:
                running = False

        if len(blocks):
            # Store the remaining blocks
            log.info('Store the remaining blocks as a new access unit')
            gvc.common.store_access_unit(output_f, acc_unit_id, acc_unit_param_set, blocks)

def run_with_threads(
    input_f,
    output_fpath,
    binarizer,
    encoder,
    axis,
    sort_rows,
    sort_cols,
    transpose,
    block_size,
    max_cols,
    dist,
    preset_mode,
    num_threads):

    log.info('run with multithreading')
    # Set two queue for workers
    queue_reader = mp.Queue()
    queue_encoder = mp.Queue()

    # Set locks for queue
    lock_reader = mp.Lock()
    lock_encoder = mp.Lock()

    process_list = []
    # set one worker to read file
    # set one worker to write file
    # the other worker do encoder job
    num_encoder = num_threads - 2

    p1 = mp.Process(target=worker_reader, name="WorkerReader", args=(input_f,block_size,queue_reader,lock_reader,num_encoder))
    process_list.append(p1)

    for i in range(num_encoder):
        p2 = mp.Process(target=worker_encoder, name="WorkerEncoder{0}".format(i),
            args=(
                binarizer,
                encoder,
                axis,
                sort_rows,
                sort_cols,
                transpose,
                block_size,
                max_cols,
                dist,
                preset_mode,
                queue_reader,
                queue_encoder,
                lock_reader,
                lock_encoder, 
            )
        )
        process_list.append(p2)

    p3 = mp.Process(target=worker_writer, name="WorkerWriter",args=(
        output_fpath,
        binarizer,
        encoder,
        axis,
        sort_rows,
        sort_cols,
        transpose,
        block_size,
        max_cols,
        dist,
        preset_mode,
        queue_encoder,
        lock_encoder,
        num_encoder,
        )
    )
    process_list.append(p3)

    for process in process_list:
        process.start()
    
    for process in process_list:
        process.join()

