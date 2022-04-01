module Fingerprint

export fingerprint, printdiff, printcompare

using ..Properties, ..JuntaSearch, RandomNumbers
using Distributions: Bernoulli

function hypercode(o, setbits = 16, length = 512)
    localrng = Xorshifts.Xorshift64(hash(o))
    return rbitvec(length, setbits, localrng)
end

@inline hypersum(a, b) = a .| b
@inline hyperdif(a, b) = a .& .~b

similarity(a, b) = sum(hypersum(a, b))

function fingerprint(t :: Vector{Tuple{BitVector, Integer, Bool}},
    codesize = nothing)

    # for Rina Panigrahy modules approach below
    # modules = [rand(Bernoulli(0.5), d, d) for _ in 1:3]

    logloglen = log(2, length(t))
    occtarget = 2
    if codesize == nothing
        codesize = max(1024, ceil(Int, logloglen))
    end

    code = BitVector(zeros(Bool, codesize))

    for entry in t
        hashinput = copy(entry[1])
        point = entry[1]
        index = entry[2]
        result = entry[3]

        # max number of bits needed to represent indices in point
        bitbound = floor(Int, log(2, length(point))) + 1

        # append index bits to hash input
        for i in 1:bitbound
            push!(hashinput, (index >> (i-1)) & 1)
        end

        # append result bit to hash input
        push!(hashinput, result)

        # set the bits of the row's code after clearing equal # of random bits
        pointcode = hypercode(hashinput, occtarget, codesize)
        pointclear = hypercode(rand(Int), occtarget, codesize)

        code = hyperdif(code, pointclear)
        code = hypersum(code, pointcode)
    end

    return code
end

@inline printdiff(a, b) = length(a) - similarity(a, b)

function crosstest(f :: Function,
    t1 :: PointwisePropertyTest, t2 :: PointwisePropertyTest)
    # Test f with t1's test on the points in t2 and collect new log;
    # return log with new results
    log = Vector{Tuple{BitVector, Integer, Bool}}()

    for pointresult in t2.log
        point = pointresult[1]
        diffbit = pointresult[2]
        t1result = t1.test(f,
            point, setindex!(copy(point), !point[diffbit], diffbit))
        push!(log, (point, diffbit, t1result))
    end

    return log
end

function printcompare(
    f1 :: Function, f2 :: Function,
    t1 :: PointwisePropertyTest, t2 :: PointwisePropertyTest,
    codesize = nothing
    )

    fullt1log = vcat(
        t1.log,
        crosstest(f1, t1, t2))
    fullt2log = vcat(
        t2.log,
        crosstest(f2, t2, t1)
    )

    return printdiff(
        fingerprint(fullt1log, codesize),
        fingerprint(fullt2log, codesize)
    )
end

end # module
