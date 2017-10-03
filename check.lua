--[[
This file samples characters from a trained model
Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.OneHot'
require 'util.misc'
require 'json'
require 'math'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-sequence',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',0,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-overall',0,'output overall mean log probability only')
cmd:option('-log',1,'output log of probability')
cmd:option('-fromfile',1,'read input sequence from file')
cmd:option('-filename','sequence.txt')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')

cmd:text()

-- parse input params
opt = cmd:parse(arg)


if opt.fromfile == 1 then
    local open = io.open
    local function read_file(path)
        local file = open(path, "rb") -- r read mode and b binary mode
        if not file then return nil end
        local content = file:read "*a" -- *a or *all reads the whole file
        file:close()
        return content
    end
    local fileContent = read_file(opt.filename);
    opt.sequence = fileContent
end



-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end


local writeFile = function(name, data)
    local path = system.pathForFile(name, system.DocumentsDirectory)
    local handle = io.open(path, "w+")
    if handle then
        handle:write(json.encode(data))
        io.close(handle)
    end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if ok and ok2 then
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if ok and ok2 then
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end
local char_log_prob = checkpoint.char_log_prob

-- initialize the rnn state to all zeros
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state

-- keep the probability of the sampled text
local sample_log_prob = nil
local sample_log_prob_before = 0
local i
probs = {}

-- do a few seeded timesteps
local seed_text = opt.sequence
if string.len(seed_text) > 0 then
    i = 1
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        -- initialize the sample probability to the empirical log probability of the first character
        if not sample_log_prob then
            sample_log_prob = char_log_prob[c]
        else
            sample_log_prob_before = sample_log_prob
            -- use the previous prediction to find the probability of this character
            sample_log_prob = sample_log_prob + prediction[1][vocab[c]]
        end
       -- io.write(ivocab[prev_char[1]])
       -- gprint(',' .. sample_log_prob - sample_log_prob_before)
        if opt.log == 0 then
        	table.insert(probs, sample_log_prob - sample_log_prob_before)
        else
        	table.insert(probs, math.exp(sample_log_prob - sample_log_prob_before))
        end
        if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
        if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
else
    -- fill with uniform probabilities over characters (? hmm)
    prediction = torch.log(torch.Tensor(1, #ivocab):fill(1)/(#ivocab))
    if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
end

-- start sampling/argmaxing
for i=1, opt.length do

    -- log probabilities from the previous timestep
    if opt.sample == 0 then
        -- use argmax
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
        prev_char_log_prob = prediction[1][prev_char[1]]
    else
        -- use sampling
        prediction:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
        prev_char_log_prob = torch.log(probs[prev_char[1]]) -- use log of probs to account for temperature effect
    end

    -- initialize the sample_log_prob (in case there was no seed text) to the emprical log prob of the selected char
    if not sample_log_prob then
        sample_log_prob = char_log_prob[ivocab[prev_char[1]]]
    else
        sample_log_prob = sample_log_prob + prev_char_log_prob
    end

    -- forward the rnn for next character
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    prediction = lst[#lst] -- last element holds the log probabilities

    io.write(ivocab[prev_char[1]])
end

if opt.overall == 1 then
	if opt.log == 1 then
		gprint('Mean sample probability: ' .. math.exp(sample_log_prob/string.len(seed_text)))
	else
		gprint('Mean sample log probability: ' .. sample_log_prob/string.len(seed_text))
	end
else
	local out_txt = {
	    { pca=probs, seq=seed_text}
	}
	local serializedJSON = json.encode( out_txt )
	local serializedJSON_2 = serializedJSON:sub(2, -2)
	print( serializedJSON_2 )
end
