module Fingerprint

export fingerprint, fingerdiff

using BitTools, ..Properties, ..JuntaSearch

function fingerprint(t :: PointwisePropertyTest)
    logloglen = log(2, length(t.log))
    occtarget = 2
    codesize = max(64, Int(ceil(logloglen)))
    return mapreduce(o -> hypercode(o, occtarget, codesize), hypersum, t.log)
end

@inline fingerdiff(a, b) = distance(a, b)

end # module
