import Foundation
import XGrammar

/// Loads HF `tokenizer.json` vocab into an xgrammar ``TokenizerInfo`` (byte-level).
public enum XGrammarTokenizer {
    public static func load(modelDir: String, eosTokenId: Int = 151_645) throws -> TokenizerInfo {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let model = root?["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Any]
        else {
            throw GuideError.badTokenizer("missing model.vocab in tokenizer.json")
        }
        var encoded = Array(repeating: "", count: vocab.count)
        var maxId = -1
        for (token, rawId) in vocab {
            let id: Int
            if let i = rawId as? Int {
                id = i
            } else if let n = rawId as? NSNumber {
                id = n.intValue
            } else {
                continue
            }
            if id < 0 { continue }
            if id >= encoded.count {
                encoded.append(contentsOf: Array(repeating: "", count: id - encoded.count + 1))
            }
            encoded[id] = token
            maxId = max(maxId, id)
        }
        if maxId + 1 < encoded.count {
            encoded = Array(encoded.prefix(maxId + 1))
        }
        // Fill holes so xgrammar sees a dense table.
        for i in encoded.indices where encoded[i].isEmpty {
            encoded[i] = "<unused_\(i)>"
        }
        return try TokenizerInfo(
            encodedVocab: encoded,
            encoding: .byteLevel,
            vocabularySize: encoded.count,
            stopTokenIDs: [Int32(eosTokenId)],
            addPrefixSpace: false)
    }

    public enum GuideError: Error, CustomStringConvertible {
        case badTokenizer(String)
        public var description: String {
            switch self {
            case .badTokenizer(let s): return s
            }
        }
    }
}

/// Stateful xgrammar matcher → allowed-next token sets (replay prefix each call).
public final class XGrammarTokenGuide: @unchecked Sendable {
    private let compiled: Grammar.Compiled
    private let vocabSize: Int
    private let eosTokenId: Int

    public init(compiled: Grammar.Compiled, vocabSize: Int, eosTokenId: Int) {
        self.compiled = compiled
        self.vocabSize = vocabSize
        self.eosTokenId = eosTokenId
    }

    public static func compileJSONSchema(
        _ schemaJSON: String,
        tokenizerInfo: TokenizerInfo,
        eosTokenId: Int
    ) async throws -> XGrammarTokenGuide {
        let grammar = Grammar(jsonSchema: schemaJSON, formatting: .compact, strictMode: true)
        let compiled = await grammar.compiled(for: tokenizerInfo)
        return XGrammarTokenGuide(
            compiled: compiled,
            vocabSize: tokenizerInfo.vocabulary.size,
            eosTokenId: eosTokenId)
    }

    public static func compileEBNF(
        _ ebnf: String,
        tokenizerInfo: TokenizerInfo,
        eosTokenId: Int
    ) async throws -> XGrammarTokenGuide {
        let grammar = Grammar(ebnf: ebnf)
        let compiled = await grammar.compiled(for: tokenizerInfo)
        return XGrammarTokenGuide(
            compiled: compiled,
            vocabSize: tokenizerInfo.vocabulary.size,
            eosTokenId: eosTokenId)
    }

    public func allowedNext(prefix: [Int]) throws -> Set<Int> {
        let matcher = try Grammar.Matcher(
            compiled,
            stopTokens: [Int32(eosTokenId)],
            terminatesWithoutStopToken: true)
        for t in prefix {
            if matcher.isTerminated { break }
            _ = matcher.accept(Int32(t))
        }
        if matcher.isTerminated {
            return [eosTokenId]
        }
        var bitmask = Grammar.Matcher.TokenBitmask(vocabSize: vocabSize)
        _ = matcher.fillNextTokenBitmask(&bitmask)
        var allowed = Set<Int>()
        allowed.reserveCapacity(64)
        for id in 0 ..< vocabSize where bitmask.isTokenAllowed(id) {
            allowed.insert(id)
        }
        if allowed.isEmpty {
            allowed.insert(eosTokenId)
        }
        return allowed
    }
}
